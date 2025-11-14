#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/training/train.py
# Copyright 2024 HuggingFace, NUS Show Lab.
# licensed under Apache License, Version 2.0 (the "License");

import os
import sys
import math
import glob
import json
import time
import shutil
import datetime
from typing import Union
from pathlib import Path
from contextlib import contextmanager

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType, set_seed
import torch

# TODO: move train module to the root dir can remove this explicit path insert
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from lightning.pytorch.utilities import CombinedLoader
from data.masking import mask_or_random_replace_tokens

from models import UniGen, get_mask_chedule
from models.model_registry import model_from_name, get_model_creator
from training.prompting_utils import UniversalPromptingQwen2, create_attention_mask_predict_next, create_attention_mask_for_mmu
from models.lr_schedulers import get_scheduler

from data.llava.llava_data_vq_unified import get_instruct_data_loader
from training.data_loader import Text2ImageDataset
from training.data_loader import make_pretrain_lm_dataloader
from utils.configuration import flatten_config, initialize_config
import utils.logger as logger_utils
from utils.checkpoint import save_checkpoint
import components.core as core


logger = core.get_logger()


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = initialize_config()
    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if config.get("ds_config", None):
        ds_plugin = DeepSpeedPlugin(hf_ds_config=config.get("ds_config"))
    else:
        ds_plugin = None
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    # wait until data fetching completed
    local_fs = Path(config.data.local_fs)
    ds_dir = (local_fs / "datasets").as_posix()
    pretrained_ckpt_dir = local_fs / "checkpoints"

    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
        (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
        * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
            (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] = config.optimizer.params.get('gradient_clipping', 'auto')

    #####################################
    # SETUP WANDB, Random seed          #
    #####################################
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    task_name = config.experiment.get('name', os.path.split(config.experiment.output_dir)[-1])
    task_name += f'-{datetime.datetime.now().strftime("%m%d%H%M")}'
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id
        wandb_init_kwargs = dict(
            name=task_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_config(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")
        logger.info(f"wandb config: {wandb_config}")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_local_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")


    config.model.unigen.llm_model_path = (pretrained_ckpt_dir / config.model.unigen.llm_model_path).as_posix()
    model_version = "qwen_2.5"
    max_len_mode = config.model.get("max_len_mode", 'text')
    model_max_length=config.model.unigen.get("model_max_length", 32768) # follow LLaVA-Onevision
    
    # initialize tokenizer for UniGen
    tokenizer = AutoTokenizer.from_pretrained(config.model.unigen.llm_model_path, model_max_length=model_max_length, padding_side="right")
    uni_prompting = UniversalPromptingQwen2(
        tokenizer,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        max_seq_len=(
            config.dataset.preprocessing.max_seq_length + config.model.unigen.num_vq_tokens + 3
            if max_len_mode == 'text' else model_max_length
        ),
        enable_reuse_tk=config.model.get("enable_reuse_tk", False), # reuse similar tokens in Qwen2 template, e.g., <|vision_start|>, <|vision_end|>
        task_token_first=config.model.get("task_token_first", True),
        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob
    )

    # VQ model for processing image into discrete tokens
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = model_from_name(config.model.vq_model.type).to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = get_model_creator(config.model.vq_model.type).from_pretrained(
            pretrained_ckpt_dir / config.model.vq_model.vq_model_name
        ).to(accelerator.device)

    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Unigen model
    use_gen_projector = config.model.unigen.get('gen_proj_depth', 0) > 0
    config.model.unigen.vocab_size = len(uni_prompting.text_tokenizer) + config.model.unigen.codebook_size + 1
    config.model.unigen.llm_vocab_size = uni_prompting.text_tokenizer.vocab_size
    config.model.unigen.num_new_special_tokens = len(uni_prompting.text_tokenizer) -  config.model.unigen.llm_vocab_size

    ##################################
    #    MODEL RESUME and LOADING   #
    #################################
    global_step = 0
    first_epoch = 0
    resume_with_accelerator = False
    accelerator.wait_for_everyone()
    if config.experiment.resume_from_checkpoint:
        resume_dir = config.experiment.output_dir
        if resume_dir is not None:
            os.makedirs(resume_dir, exist_ok=True)
        dirs = os.listdir(resume_dir)
        logger.info(f"Checkpoints in current exp dir: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        reusme_path = dirs[-1] if len(dirs) > 0 else None
        if reusme_path is not None:
            try:
                reusme_path = os.path.join(resume_dir, reusme_path)
                global_step = int(os.path.basename(reusme_path).split("-")[1])
                logger.info(f"Resuming from checkpoint {reusme_path}")
                if os.path.exists(os.path.join(reusme_path, 'unwrapped_model')):
                    config.model.unigen.pretrained_model_path = os.path.join(reusme_path, 'unwrapped_model')
                    config.model.unigen.load_from_pretrained = True
                else:
                    resume_with_accelerator = True
            except:
                logger.info(f"No available checkpoint for resuming")

    if config.model.unigen.load_from_pretrained:
        pretrained_model_path = (pretrained_ckpt_dir / config.model.unigen.pretrained_model_path).as_posix()
        use_safetensors = config.model.get('load_with_safetensors', None)
        if not pretrained_model_path.endswith("unwrapped_model"):
            if os.path.exists(os.path.join(pretrained_model_path, 'unwrapped_model')):
                 pretrained_model_path = os.path.join(pretrained_model_path, 'unwrapped_model')
            else:
                ckpt_files = glob.glob(os.path.join(pretrained_model_path, "*/unwrapped_model"))
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
                pretrained_model_path = ckpt_files[0]
        if len(glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))) > 0:
            use_safetensors = True
        elif len(glob.glob(os.path.join(pretrained_model_path, "pytorch_model*.bin"))) > 0:
            use_safetensors = False
        model, msg = UniGen.from_pretrained(
            pretrained_model_path,
            use_safetensors=use_safetensors,
            ckpt_base_path = pretrained_ckpt_dir.as_posix(),
            output_loading_info=True)
        logger.info(f"load from {pretrained_model_path} use_safetensors: {use_safetensors} {msg}")
        if model.llm.config.vocab_size != config.model.unigen.vocab_size:
            model.llm.resize_token_embeddings(config.model.unigen.vocab_size)
            model.register_to_config(vocab_size=config.model.unigen.vocab_size)
            model.llm.config.vocab_size = config.model.unigen.vocab_size
            model.output_size = config.model.unigen.vocab_size
            if not use_gen_projector:
                model.register_to_config(mask_token_id=model.config.vocab_size - 1)
            else:
                model.register_to_config(mask_token_id=config.codebook_size)
    else:
        model = UniGen(**config.model.unigen)
    model.to(accelerator.device)
    model.llm.config.use_cache = False
    logger.info(f"model config: {model.config}, {model.llm.config}")
    mask_id = model.mask_token_id

    if config.model.get("mm_tunable_parts", None) is not None:
        tunable_parts = config.model.mm_tunable_parts.split(",")
        model.requires_grad_(False)
        if hasattr(model, 'module'):
            for n, p in model.module.named_parameters():
                for tp in tunable_parts:
                    if tp in n:
                        p.requires_grad = True
        else:
            for n, p in model.named_parameters():
                for tp in tunable_parts:
                    if tp in n:
                        p.requires_grad = True
        if "mm_new_embed" in tunable_parts:
            embedding_layer = model.module.llm.model.embed_tokens if hasattr(model, 'module') else model.llm.model.embed_tokens
            lm_head = model.module.llm.lm_head if hasattr(model, 'module') else model.llm.lm_head
            embedding_layer.requires_grad_(True)
            lm_head.requires_grad_(True)

    use_causal_mask  = config.model.get('use_causal_mask', False)
    t2i_gen_mode = config.model.get('t2i_gen_mode', 'mask')
    assert use_causal_mask == True or t2i_gen_mode == 'mask'
    
    ##################################
    #   Optimizer #
    #################################
    optimizer_config = config.optimizer.params
    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")
    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.training.batch_size_t2i > 0:
        if config.dataset.gen_type == "t2i":
            dataset = Text2ImageDataset(
                train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
                tokenizer=None,  # we want to get raw texts
                max_seq_length=preproc_config.max_seq_length,
                num_train_examples=config.experiment.max_train_examples_t2i,
                per_gpu_batch_size=config.training.batch_size_t2i,
                global_batch_size=total_batch_size_t2i_without_accum,
                num_workers=dataset_config.num_workers,
                resolution=preproc_config.resolution,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size,
                pin_memory=dataset_config.pin_memory,
                persistent_workers=dataset_config.persistent_workers,
                short_caption_ratio=dataset_config.get('t2i_short_caption_ratio', 0.5),
                data_dir=ds_dir
            )
            train_dataloader_t2i = dataset.train_dataloader
            t2i_update_steps_per_epoch = math.ceil(
                train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
            logger.info(f"number of t2i dataset update per epoch: {t2i_update_steps_per_epoch}")
        
        else:
            raise ValueError(f"Unsupported dataset type {config.dataset.type}")

    total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes
    # Data for image captioning
    if config.training.batch_size_mmu > 0:
        # remove undefined dataset captioning_parquet
        if config.dataset.und_type == "captioning":
            dataset_mmu = Text2ImageDataset(
                train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
                tokenizer=None,  # we want to get raw texts
                max_seq_length=preproc_config.max_seq_length,
                num_train_examples=config.experiment.max_train_examples_mmu,
                per_gpu_batch_size=config.training.batch_size_mmu,
                global_batch_size=total_batch_size_mmu_without_accum,
                num_workers=dataset_config.num_workers,
                resolution=preproc_config.resolution,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size,
                pin_memory=dataset_config.pin_memory,
                persistent_workers=dataset_config.persistent_workers,
                is_captioning=True,
                add_caption_prompt=dataset_config.add_caption_prompt,
                short_caption_ratio=dataset_config.get('mmu_short_caption_ratio', 0.),
                caption_file=dataset_config.get('caption_file', 'data/prompts/short_caption_prompt.json'),
                data_dir=ds_dir
            )
            train_dataloader_mmu = dataset_mmu.train_dataloader
            mmu_update_steps_per_epoch = math.ceil(train_dataloader_mmu.num_batches / config.training.gradient_accumulation_steps)
            logger.info(f"number of mmu dataset update per epoch: {mmu_update_steps_per_epoch}")

        elif config.dataset.und_type in ["llava_pretrain","llava_tuning"]:
            # llava_pretrain  train_mmu_shards_path_or_url: [ "llava/blip_laion_cc_sbu_558k.json", "llava/pretraining_data"]
            # llava_tuning  train_mmu_shards_path_or_url: [ "llava/llava_v1_5_mix665k.json", "llava/tuning_data"]
            # sharegpt4v_pretrain  train_mmu_shards_path_or_url: ["ShareGPT4V/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json", "ShareGPT4V/data"]
            # sharegpt4v_tuning  train_mmu_shards_path_or_url: ["ShareGPT4V/sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json", "ShareGPT4V/data"]
            llava_repeat_n = dataset_config.get('llava_repeat_n', 1)
            if isinstance(dataset_config.train_mmu_shards_path_or_url[0], str):
                llava_data_path = [dataset_config.train_mmu_shards_path_or_url[0]]  * llava_repeat_n
            else:
                llava_data_path = list(dataset_config.train_mmu_shards_path_or_url[0])  * llava_repeat_n
            train_dataloader_mmu = get_instruct_data_loader(
                tokenizer,
                batch_size=config.training.batch_size_mmu,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=preproc_config.max_seq_length, # if config.dataset.add_system_prompt else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
                add_system_prompt=config.dataset.add_system_prompt,
                resolution=preproc_config.resolution,
                data_path=[os.path.join(ds_dir, data_path) for data_path in llava_data_path],
                image_root=os.path.join(ds_dir, dataset_config.train_mmu_shards_path_or_url[1]),
                disable_text_rich=dataset_config.get('disable_text_rich', False),
            )
            mmu_update_steps_per_epoch = math.ceil(len(train_dataloader_mmu) / config.training.gradient_accumulation_steps)
            logger.info(f"number of mmu dataset update per epoch: {mmu_update_steps_per_epoch}")

        else:
            raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # LLM pure text dataset: RefinedWeb
    if config.training.batch_size_lm > 0:
        if isinstance( dataset_config.train_lm_shards_path_or_url, str):
            train_lm_shards_path_or_url = [os.path.join(ds_dir, dataset_config.train_lm_shards_path_or_url)]
        else:
            train_lm_shards_path_or_url = [os.path.join(ds_dir, data_path) for data_path in dataset_config.train_lm_shards_path_or_url]
        train_dataloader_lm = make_pretrain_lm_dataloader(train_lm_shards_path_or_url,
                tokenizer=tokenizer,
                batch_size=config.training.batch_size_lm,
                num_workers=dataset_config.num_workers,
                world_size=accelerator.num_processes,
                local_rank=accelerator.process_index,
                max_length=uni_prompting.max_seq_len,
                repeat_n=dataset_config.get('lm_repeat', 1))
        lm_update_steps_per_epoch = math.ceil(len(train_dataloader_lm) / config.training.gradient_accumulation_steps)
        logger.info(f"number of lm dataset update per epoch: {lm_update_steps_per_epoch}")

    # Combine these dataloaders into a single iterable model
    iterables = dict()
    if config.training.batch_size_lm > 0:
        iterables["lm_flow"] = train_dataloader_lm
        
    if config.training.batch_size_mmu > 0:
        iterables["mmu_flow"] = train_dataloader_mmu
        
    if config.training.batch_size_t2i > 0:
        iterables["t2i_flow"] = train_dataloader_t2i
    
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)
    
    #TODO considering training epochs
    if config.training.batch_size_mmu > 0 and config.training.batch_size_t2i > 0:
        num_update_steps_per_epoch = min(mmu_update_steps_per_epoch, t2i_update_steps_per_epoch)
    elif config.training.batch_size_mmu > 0:
        num_update_steps_per_epoch = mmu_update_steps_per_epoch
    elif config.training.batch_size_t2i > 0:
        num_update_steps_per_epoch = t2i_update_steps_per_epoch
    else:
        num_update_steps_per_epoch = lm_update_steps_per_epoch

    ##################################
    #  Lr scheduler #
    #################################
    num_train_epochs = config.training.get('num_train_epochs', 1)
    if config.training.max_train_steps == -1:
        config.training.max_train_steps = int(math.ceil(num_update_steps_per_epoch * num_train_epochs / 500.) * 500)

    min_lr_scale = float(config.lr_scheduler.params.get('min_lr', 0) ) /  optimizer_config.learning_rate
    if accelerate.__version__ < '1.0.0':
        lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=optimizer,
            num_training_steps=config.training.max_train_steps,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps,
            min_scale = min_lr_scale,
        )
    else:
        lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            min_scale = min_lr_scale,
        )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if reusme_path is not None and resume_with_accelerator:
        accelerator.load_state(reusme_path)

    vq_model.to(device=accelerator.device)

    if use_causal_mask:
        mask_dtype = torch.bool
    elif hasattr(model, 'module'):
        mask_dtype = model.module.llm.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.llm.model.embed_tokens.weight.dtype
    logger.info(f"weight_dtype: {mask_dtype}")


    ##################################
    #             Training          #
    #################################
    generate_every_pred = config.experiment.get('generate_every_pred', 0)
    eval_batch_size = config.experiment.get('eval_batch_size', 16)
    eval_text_len = config.experiment.get('eval_text_len', 128)

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Start training global step = {global_step}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
            use_gen_projector: bool = False,
            t2i_gen_mode='mask',
    ):

        image_tokens = vq_model.get_code(pixel_values_or_image_ids)
        if not use_gen_projector:
            image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

        # create MLM mask and labels
        if t2i_gen_mode == 'mask':
            input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
                image_tokens,
                mask_id,
                config,
                mask_schedule=mask_schedule,
                is_train=is_train,
            )
        else:
            input_ids = image_tokens
            labels = input_ids
            mask_prob =  torch.tensor(0.).to(input_ids.device)
            
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')
        return input_ids, masks, labels, mask_prob, image_tokens

    batch_time_m = logger_utils.AverageMeter()
    data_time_m = logger_utils.AverageMeter()
    fw_time_m = logger_utils.AverageMeter()
    bw_time_m = logger_utils.AverageMeter()
    loss_t2i_m =  logger_utils.AverageMeter()
    loss_mmu_m =  logger_utils.AverageMeter()
    loss_lm_m =  logger_utils.AverageMeter()
    end = time.time()
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for idx, (batch, batch_idx, dataloader_idx) in enumerate(tqdm(iter(combined_dataloader))):
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]  if config.training.batch_size_t2i > 0 else 0
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0]  if config.training.batch_size_mmu > 0 else 0
            batch_size_lm = len(batch["lm_flow"]["input_ids"]) if config.training.batch_size_lm > 0 else 0
        
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Encode images to image tokens, mask them and create input and labels
            if batch_size_t2i > 0:
                pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
                t2i_soucrce = batch["t2i_flow"]["source"] if "source" in batch["t2i_flow"] else None
                pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
                data_time_m.update(time.time() - end)
                # Encode images to image tokens, mask them and create input and labels
                (
                    input_ids,
                    attention_mask_t2i,
                    labels,
                    mask_prob,
                    image_tokens_ori
                ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate,
                                              use_gen_projector=use_gen_projector,
                                              t2i_gen_mode=t2i_gen_mode)

                if use_causal_mask:
                    attention_mask = attention_mask_t2i
                else:
                    attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True,
                                                                    return_inverse_mask=True)
                attention_mask = attention_mask.to(mask_dtype)
                if use_gen_projector:
                    soi_pos = -config.model.unigen.num_vq_tokens - 2 if input_ids[0,-1] == uni_prompting.eos_token_id else  - config.model.unigen.num_vq_tokens - 1
                    torch.where(input_ids == int(uni_prompting.sptids_dict['<|soi|>']))[1][0]
                    if hasattr(model, 'module'):
                        input_part1 =  model.module.llm.model.embed_tokens(input_ids[:, :soi_pos])
                        input_image = model.module.get_gen_embed(input_ids[:, soi_pos: soi_pos + config.model.unigen.num_vq_tokens].contiguous())
                        input_part2 = model.module.llm.model.embed_tokens(input_ids[:, soi_pos + config.model.unigen.num_vq_tokens: ])
                        text_embeddings_img_text = torch.cat([input_part1, input_image, input_part2], 1)
                    else:
                        input_part1 =  model.llm.model.embed_tokens(input_ids[:, :soi_pos])
                        input_image = model.get_gen_embed(input_ids[:, soi_pos : soi_pos + config.model.unigen.num_vq_tokens])
                        input_part2 = model.llm.model.embed_tokens(input_ids[:, soi_pos + config.model.unigen.num_vq_tokens: ])
                        text_embeddings_img_text = torch.cat([input_part1, input_image, input_part2], 1)
            else:
                mask_prob = torch.tensor(0.).to(accelerator.device)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for language modeling
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if batch_size_lm > 0:
                texts_lm = batch["lm_flow"]["input_ids"]
                if batch_size_t2i > 0:
                    input_ids_lm, attention_mask_lm, labels_lm = uni_prompting((texts_lm, input_ids.shape[-1]), 'lm')
                else:
                    input_ids_lm, attention_mask_lm, labels_lm = uni_prompting((texts_lm, uni_prompting.max_seq_len), 'lm')
                if not use_causal_mask:
                    attention_mask_lm = create_attention_mask_predict_next(
                        input_ids_lm.to(input_ids.device),
                        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])
                    )
                attention_mask_lm = attention_mask_lm.to(mask_dtype).to(attention_mask.device)
                if use_gen_projector:
                    if hasattr(model, 'module'):
                        text_embeddings_lm = model.module.llm.model.embed_tokens(input_ids_lm.to(accelerator.device))
                    else:
                        text_embeddings_lm = model.llm.model.embed_tokens(input_ids_lm.to(accelerator.device))
                if batch_size_t2i > 0:
                    attention_mask = torch.cat([attention_mask, attention_mask_lm], dim=0)
                    input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
                    labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)
                    if use_gen_projector:
                        text_embeddings_img_text = torch.cat([text_embeddings_img_text, text_embeddings_lm])
                else:
                    input_ids = input_ids_lm.to(accelerator.device)
                    attention_mask = attention_mask_lm
                    input_ids = input_ids_lm.to(accelerator.device)
                    labels = labels_lm.to(accelerator.device)
                    if use_gen_projector:
                        text_embeddings_img_text = text_embeddings_lm

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for captioning/multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if batch_size_mmu > 0:
                if "llava" in config.dataset.und_type or "sharegpt4v" in config.dataset.und_type:
                    pixel_values_mmu, input_ids_mmu, labels_mmu = (
                        batch["mmu_flow"]["images"],
                        batch["mmu_flow"]["input_ids"],
                        batch["mmu_flow"]["labels"]
                    )
                    pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                    input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
                    image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                    image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                    if use_gen_projector:
                        if hasattr(model, 'module'):
                            images_embeddings = model.module.get_gen_embed(image_tokens_mmu)
                        else:
                            images_embeddings = model.get_gen_embed(image_tokens_mmu)
                        input_ids_part1, input_ids_part2, attention_mask_mmu, labels_mmu= uni_prompting((images_embeddings, input_ids_mmu, labels_mmu, None), 'mmu_conv')
                        if hasattr(model, 'module'):
                            part1 = model.module.llm.model.embed_tokens(input_ids_part1)
                            part2 = model.module.llm.model.embed_tokens(input_ids_part2)
                        else:
                            part1 = model.llm.model.embed_tokens(input_ids_part1)
                            part2 = model.llm.model.embed_tokens(input_ids_part2)
                        input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
                    else:
                        input_ids_mmu, attention_mask_mmu, labels_mmu = uni_prompting((image_tokens_mmu, input_ids_mmu, labels_mmu, None), 'mmu_conv')

                else:
                    pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
                    pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                    image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                    image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                    input_ids_mmu, attention_mask_mmu, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), 'mmu')
                    input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
                    if use_gen_projector:
                        input_ids_part1, input_ids_part2, attention_mask_mmu, labels_mmu= uni_prompting((image_tokens_mmu, input_ids_mmu), 'mmu_emb')
                        if hasattr(model, 'module'):
                            images_embeddings = model.module.get_gen_embed(image_tokens_mmu)
                            part1 = model.module.llm.model.embed_tokens(input_ids_part1)
                            part2 = model.module.llm.model.embed_tokens(input_ids_part2)
                        else:
                            images_embeddings = model.get_gen_embed(image_tokens_mmu)
                            part1 = model.llm.model.embed_tokens(input_ids_part1)
                            part2 = model.llm.model.embed_tokens(input_ids_part2)
                        input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)

                if not use_causal_mask:
                    attention_mask_mmu = create_attention_mask_for_mmu(input_ids_mmu.to(accelerator.device), eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                attention_mask_mmu = attention_mask_mmu.to(mask_dtype)
                if batch_size_lm > 0 or batch_size_t2i  > 0:
                    attention_mask = torch.cat([attention_mask, attention_mask_mmu], dim=0)
                    labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)
                    if use_gen_projector:
                        input_embeddings = torch.cat([text_embeddings_img_text, input_embeddings], dim=0)
                    else:
                        input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
                else:
                    attention_mask = attention_mask_mmu
                    labels = labels_mmu.to(input_ids.device)
                    input_ids = input_ids_mmu.to(attention_mask.device)
                    if use_gen_projector:
                        input_embeddings =  text_embeddings_img_text

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))
            with accelerator.accumulate(model):
                fw_start =time.time()
                if use_gen_projector:
                    logits, loss_t2i, loss_lm, loss_mmu = model(
                    input_ids=input_ids,
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                    labels=labels,
                    label_smoothing=config.training.label_smoothing,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    num_vq_tokens=config.model.unigen.num_vq_tokens,
                    t2i_mode=t2i_gen_mode,
                )
                else:
                    logits, loss_t2i, loss_lm, loss_mmu = model(
                        input_ids=input_ids,
                        input_embeddings=None,
                        attention_mask=attention_mask,
                        labels=labels,
                        label_smoothing=config.training.label_smoothing,
                        batch_size_t2i=batch_size_t2i,
                        batch_size_lm=batch_size_lm,
                        batch_size_mmu=batch_size_mmu,
                        max_seq_length=config.dataset.preprocessing.max_seq_length,
                        num_vq_tokens=config.model.unigen.num_vq_tokens,
                        t2i_mode=t2i_gen_mode,
                    )
                fw_time_m.update(time.time() - fw_start)

                # Gather the losses across all processes for logging (if we use distributed training).
                bw_start = time.time()

                if batch_size_t2i == 0:
                    loss_t2i = torch.tensor(0.).to(input_ids.device)
                else:
                    loss_t2i_m.update(loss_t2i)
                if batch_size_lm == 0:
                    loss_lm = torch.tensor(0.).to(input_ids.device)
                else:
                    loss_lm_m.update(loss_lm)
                if batch_size_mmu == 0:
                    loss_mmu = torch.tensor(0.).to(input_ids.device)
                else:
                    loss_mmu_m.update(loss_mmu)

                loss = (
                    config.training.t2i_coeff * loss_t2i
                    + config.training.lm_coeff * loss_lm
                    + config.training.mmu_coeff * loss_mmu
                )

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                bw_time_m.update(time.time() - bw_start)

                # log gradient norm before zeroing it
                should_log = (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                )
                if should_log:
                    logger_utils.log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    avg_loss_lm = loss_lm_m.avg if loss_lm_m.count > 0 else torch.tensor(0.)
                    avg_loss_t2i= loss_t2i_m.avg  if loss_t2i_m.count > 0 else torch.tensor(0.)
                    avg_loss_mmu = loss_mmu_m.avg  if loss_mmu_m.count > 0 else torch.tensor(0.)
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.avg,
                        "batch_time": batch_time_m.avg,
                        "forward_time": fw_time_m.avg,
                        "backward_time": bw_time_m.avg
                    }
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                    fw_time_m.reset()
                    bw_time_m.reset()
                    loss_lm_m.reset()
                    loss_mmu_m.reset()
                    loss_t2i_m.reset()

                # Save model checkpoint
                if global_step +  1 < config.training.max_train_steps:
                    if (global_step + 1) % config.experiment.save_every == 0:
                        accelerator.wait_for_everyone()
                        save_checkpoint(model, config, accelerator, global_step + 1, save_last=False)

                    #We stop evaluatuon for FSDP-job to avoid unsharding problems
                    if batch_size_t2i > 0 and (global_step + 1) % config.experiment.generate_every == 0:
                        generate_images(
                            model,
                            vq_model,
                            uni_prompting,
                            accelerator,
                            config,
                            global_step + 1,
                            mask_schedule=mask_schedule,
                            batch_size=eval_batch_size,
                            text_len=eval_text_len
                        )
                    if generate_every_pred > 0  and batch_size_t2i > 0 and (global_step + 1) % generate_every_pred == 0:
                        visualize_predictions(
                            model,
                            accelerator,
                            vq_model,
                            uni_prompting,
                            config,
                            global_step + 1,
                            input_ids,
                            image_tokens_ori,
                            batch["t2i_flow"]["images"],
                            texts,
                            logits,
                            t2i_gen_mode,
                            data_source=t2i_soucrce,
                            use_gen_projector=use_gen_projector
                        )
                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for
        if global_step >= config.training.max_train_steps:
            break

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step, save_last=True)

    accelerator.wait_for_everyone()
    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
        model,
        accelerator,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
        t2i_gen_mode,
        data_source=None,
        use_gen_projector=False
):
    logger.info("Visualizing predictions...")
    model.eval()

    if use_gen_projector:
        recons_images = vq_model.decode_code(image_tokens_ori)
    else:
        recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    soi_pos = torch.where(input_ids == int(uni_prompting.sptids_dict['<|soi|>']))[1][0]

    if t2i_gen_mode == 'mask':
        # predictions = logits[:config.training.batch_size_t2i, -(config.model.unigen.num_vq_tokens + 1):-1:,
        #           config.model.unigen.llm_vocab_size + config.model.unigen.num_new_special_tokens:-1]
        if use_gen_projector:
            predictions = logits[:config.training.batch_size_t2i, soi_pos+ 1: soi_pos + config.model.unigen.num_vq_tokens + 1,]
            predictions = predictions.argmax(axis=-1)
            mask_token_id = config.model.unigen.codebook_size
            input_ids = input_ids[:config.training.batch_size_t2i, soi_pos+ 1: soi_pos + config.model.unigen.num_vq_tokens + 1] 
        else:
            predictions = logits[:config.training.batch_size_t2i, soi_pos+ 1: soi_pos + config.model.unigen.num_vq_tokens + 1,
                        config.model.unigen.llm_vocab_size + config.model.unigen.num_new_special_tokens:-1]
            predictions = predictions.argmax(axis=-1)
            mask_token_id = config.model.unigen.vocab_size - 1 - len(uni_prompting.text_tokenizer)
            input_ids = input_ids[:config.training.batch_size_t2i, soi_pos+ 1: soi_pos + config.model.unigen.num_vq_tokens + 1] - len(uni_prompting.text_tokenizer)
        mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
            dim=-1) / config.model.unigen.num_vq_tokens).cpu().numpy())
        predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
    elif t2i_gen_mode == 'ar':
        predictions = logits[:config.training.batch_size_t2i, soi_pos: soi_pos + config.model.unigen.num_vq_tokens,
                    config.model.unigen.llm_vocab_size + config.model.unigen.num_new_special_tokens:-1]
        predicted_images = predictions.argmax(axis=-1)
        mask_ratio = [0.] * predictions.shape[0]

    predicted_images = torch.clamp(predicted_images, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    predicted_images = vq_model.decode_code(predicted_images)

    if accelerator.is_main_process:
        predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
        predicted_images *= 255.0
        predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
        pil_images = [Image.fromarray(image) for image in predicted_images]

        # Log images
        if data_source is None:
            wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
                            enumerate(zip(pil_images, mask_ratio))]
        else:
            wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n source: {data_source[i]} \n caption: {texts[i]}') for i, (image, r) in
                            enumerate(zip(pil_images, mask_ratio))]
        wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)

    model.train()

@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
        batch_size,
        text_len,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    wandb_images = []
    use_causal_mask  = config.model.get('use_causal_mask', False)
    t2i_gen_mode = config.model.get('t2i_gen_mode', 'mask')

    if use_causal_mask:
        mask_dtype = torch.bool
    elif hasattr(model, 'module'):
        mask_dtype = model.module.llm.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.llm.model.embed_tokens.weight.dtype

    if config.model.unigen.get('gen_proj_depth', 0) > 0:
        mask_token_id = config.model.unigen.codebook_size
    else:
        mask_token_id = config.model.unigen.vocab_size - 1
    for i in range(0, len(validation_prompts), batch_size):
        validation_prompts_ = validation_prompts[i: i + batch_size]
        image_tokens = torch.ones((len(validation_prompts_), config.model.unigen.num_vq_tokens), dtype=torch.long,
                                device=accelerator.device) * mask_token_id
        input_ids, attention_mask = uni_prompting((validation_prompts_, image_tokens, text_len), 't2i_gen')
        if config.training.guidance_scale > 0:
            uncond_input_ids, attention_mask_ = uni_prompting(([''] * len(validation_prompts_), image_tokens, text_len), 't2i_gen')
            if use_causal_mask:
                attention_mask = torch.cat([attention_mask, attention_mask_], dim=0).to(mask_dtype)
            else:
                attention_mask = create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True
                ).to(mask_dtype)
        else:
            if use_causal_mask:
                attention_mask = attention_mask.to(mask_dtype)
            else:
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True).to(mask_dtype
                )
            uncond_input_ids = None

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

    
        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            if t2i_gen_mode == 'mask':
                gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    text_vocab_size=len(uni_prompting.text_tokenizer),
                    image_token_num_per_image=config.model.unigen.num_vq_tokens,
                )
            elif t2i_gen_mode == 'ar':
                gen_token_ids = accelerator.unwrap_model(model).t2i_generate_ar(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    text_vocab_size=len(uni_prompting.text_tokenizer),
                    image_token_num_per_image=config.model.unigen.num_vq_tokens,
                )
            # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
            # so we clamp them to the correct range.
            gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

        # Convert to PIL images
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).to(torch.float32).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        # Log images
        wandb_images += [(image, validation_prompts_[i]) for i, image in enumerate(pil_images)]

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model
    if accelerator.is_main_process:
        wandb_images = [wandb.Image(image, caption=caption) for image, caption in wandb_images]
        wandb.log({"Generated images": wandb_images}, step=global_step)


if __name__ == "__main__":
    main()
