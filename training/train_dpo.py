#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/training/train_w_clip_vit.py
# Copyright 2024 HuggingFace, NUS Show Lab.
# licensed under Apache License, Version 2.0 (the "License");

import json
import logging
import sys
import os
import copy
import random
import math
import glob
import datetime
from contextlib import contextmanager
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.append("./")
from pathlib import Path
from typing import Union, Dict, Optional, List, Any
import numpy as np
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType, set_seed
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from data.masking import mask_or_random_replace_tokens
from dataclasses import field, dataclass
from models import UniGen, get_mask_chedule
from models.model_registry import model_from_name, get_model_creator
from training.prompting_utils import UniversalPromptingQwen2, create_attention_mask_predict_next
from models.lr_schedulers import get_scheduler
import components.core as core
from utils.configuration import flatten_config, initialize_config
import utils.logger as logger_utils
from utils.checkpoint import save_checkpoint

logger = core.get_logger()

def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        num_vq_tokens=256,
        t2i_gen_mode='mask',
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    logits = logits[:,-(num_vq_tokens + 1):-1]
    labels = labels[:,-(num_vq_tokens + 1):-1]
    labels = labels.clone()
    loss_mask = labels != label_pad_token_id # B, N

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0
    if t2i_gen_mode == 'ar':
        per_token_logps = torch.gather(logits[:, :-1].log_softmax(-1), dim=2, index=labels[:, 1:].unsqueeze(2)).squeeze(2)
        loss_mask = loss_mask[:, 1:]
    else:
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def get_vq_model_class(model_type):
    assert model_type == "magvitv2"
    return MAGVITv2


def load_data(data_path):
    data_list = []
    if "jsonl" in data_path:
        with open(data_path, "r") as json_file:
            for line in json_file:
                data_list.append(json.loads(line.strip()))
    else:
        with open(data_path, "r") as json_file:
            data_list = json.load(json_file)
    return data_list

class DPODataset(Dataset):
    def __init__(self, config: dict, data_dir:str, device=None):
        super(DPODataset, self).__init__()
        # Handle multiple JSON files specified in the data_path
        self.list_data_dict = []

        datasets = config.dataset.params.train_t2i_shards_path_or_url[0]
        if isinstance(datasets, str):
            datasets = [datasets]
        data_root = os.path.join(data_dir, config.dataset.params.train_t2i_shards_path_or_url[1])
        sampling_strategy =config.dataset.params.get("sampling_strategy", "all")
        sampling_number_list =config.dataset.params.get("sampling_number", ['100%'] * len(datasets))
        assert len(sampling_number_list) == len(datasets)
        # assert data_path.endswith(".yaml")
        # with open(data_path, "r") as file:
        #     yaml_data = yaml.safe_load(file)
        #     datasets = yaml_data.get("datasets")
        for json_path, sampling_number in zip(datasets, sampling_number_list):
            logger.info(f"Loading {json_path} with {sampling_strategy} sampling strategy")
            cur_data_dict = load_data(os.path.join(data_dir, json_path))

            if "%" in sampling_number:
                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
            else:
                sampling_number = int(sampling_number)

            # Apply the sampling strategy
            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            logger.info(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            self.list_data_dict.extend(cur_data_dict)

        self.config = config
        self.device = device
        self.data_root = data_root

    def __len__(self):
        return len(self.list_data_dict)

    def transform_image(self, image: Image):
        resolution = self.config.dataset.preprocessing.resolution
        image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.CenterCrop((resolution, resolution))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
        return image

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        data_dict = copy.deepcopy(self.list_data_dict[i])
        data_dict["chosen"] = self.transform_image(Image.open(os.path.join(self.data_root, data_dict["chosen"])))
        data_dict['rejected'] = self.transform_image(Image.open(os.path.join(self.data_root, data_dict["rejected"])))
        return data_dict

@dataclass
class DPODataCollator(object):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_prompt_list = [feature['prompt'] for feature in features]
        batch_chosen_image = torch.stack([feature['chosen'] for feature in features], dim=0)
        batch_rejected_image = torch.stack([feature['rejected'] for feature in features], dim=0)
        final_batch = {}
        final_batch['batch_prompt_list'] = batch_prompt_list
        final_batch['batch_chosen_image'] = batch_chosen_image
        final_batch['batch_rejected_image'] = batch_rejected_image
        return final_batch

def main():
    config = initialize_config()
    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    if config.get("ds_config", None):
        ds_plugin = DeepSpeedPlugin(hf_ds_config=config.get("ds_config"))
    else:
        ds_plugin = None
    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
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


    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.batch_size_t2i
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
        logger.info(f"wandb config: {wandb_config}")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    device = accelerator.device

    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    config.model.unigen.llm_model_path = (pretrained_ckpt_dir / config.model.unigen.llm_model_path).as_posix()
    max_len_mode = config.model.get("max_len_mode", 'text')
    model_max_length=config.model.unigen.get("model_max_length", 32768) # follow LLaVA-Onevision
    
    # load UniGen model
    tokenizer = AutoTokenizer.from_pretrained(config.model.unigen.llm_model_path, model_max_length=model_max_length, padding_side="right")
    uni_prompting = UniversalPromptingQwen2(
            tokenizer,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                "<|mmu|>", "<|t2v|>", "<|ti2i|>", "<|vision_sep|>"
            ),
            max_seq_len=(
                config.dataset.preprocessing.max_seq_length + config.model.unigen.num_vq_tokens + 3
                if max_len_mode == 'text' else model_max_length
            ),
            enable_reuse_tk=config.model.get("enable_reuse_tk", False), # reuse similar tokens in Qwen2 template, e.g., <|vision_start|>, <|vision_end|>
            task_token_first=config.model.get("task_token_first", True),
            ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob
        )

    logger.info(f"special tokens : {uni_prompting.sptids_dict}")

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
    vq_model.to(device=accelerator.device)

    # Initialize Unigen Model
    use_gen_projector = config.model.unigen.get('gen_proj_depth', 0) > 0
    
    config.model.unigen.vocab_size = len(uni_prompting.text_tokenizer) + config.model.unigen.codebook_size + 1
    config.model.unigen.llm_vocab_size = uni_prompting.text_tokenizer.vocab_size
    config.model.unigen.num_new_special_tokens = len(uni_prompting.text_tokenizer) -  config.model.unigen.llm_vocab_size
    if use_gen_projector:
        config.model.unigen.vocab_size -= config.model.unigen.codebook_size + 1
        
    vision_tower = model_from_name(
        (pretrained_ckpt_dir / config.model.vision_tower.name).as_posix()
    ).to(accelerator.device)
    visual_processor = vision_tower.image_processor
    vision_tower.eval()
    
    ##################################
    #         MODEL RESUME and LOADING  #
    #################################
    global_step = 0
    first_epoch = 0
    resume_with_accelerator = False
    accelerator.wait_for_everyone()
    if config.model.unigen.load_from_pretrained:
        pretrained_model_path = (pretrained_ckpt_dir / config.model.unigen.pretrained_model_path).as_posix()
        use_safetensors = config.model.get('load_with_safetensors', None)
    
        if not pretrained_model_path.endswith("unwrapped_model"):
            if os.path.exists(os.path.join(pretrained_model_path, 'unwrapped_model')):
                 pretrained_model_path = os.path.join(pretrained_model_path, 'unwrapped_model')
            else:
                ckpt_files = glob.glob(os.path.join(pretrained_model_path, "*/unwrapped_model"))
                if len(ckpt_files) > 0:
                    ckpt_files.sort(key=os.path.getmtime, reverse=True)
                    pretrained_model_path = ckpt_files[0]
        if len(glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))) > 0:
            use_safetensors = True
        elif len(glob.glob(os.path.join(pretrained_model_path, "pytorch_model*.bin"))) > 0:
            use_safetensors = False
        model, msg = UniGen.from_pretrained(
            pretrained_model_path,
            use_safetensors=use_safetensors,
            ckpt_base_path=pretrained_ckpt_dir.as_posix(),
            output_loading_info=True)
        ref_model, msg = UniGen.from_pretrained(
            pretrained_model_path,
            use_safetensors=use_safetensors,
            ckpt_base_path=pretrained_ckpt_dir.as_posix(),
            output_loading_info=True)
        logger.info(f"load from {pretrained_model_path} use_safetensors: {use_safetensors} {msg}")
        
        model.to(accelerator.device)
    else:
        model = UniGen(**config.model.unigen).to(device)
        # load_model_weights(model, config.model.unigen.pretrained_model_path, str(device))
        ref_model = UniGen(**config.model.unigen).to(device)
        # load_model_weights(ref_model, config.model.unigen.pretrained_model_path, str(device)) ###

    model.to(accelerator.device)
    ref_model.to(accelerator.device)
    model.llm.config.use_cache = False
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        print(f"Enabale gradient checkpointing in Unigen")
        model.llm.gradient_checkpointing_enable()
    logger.info(f"model config: {model.config}, {model.llm.config}")
    mask_id = model.mask_token_id

    parameter_names = [n for n, _ in ref_model.named_parameters()]
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    ref_model.eval()

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

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
      
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    train_dataset = DPODataset(config=config, device=device, data_dir=ds_dir)
    data_collator = DPODataCollator()
    dataloader_params = {
        "batch_size": config.training.batch_size_t2i,
        "collate_fn": data_collator,
        "shuffle": True,
        "num_workers": config.dataset.params.num_workers,
        "pin_memory": config.dataset.params.pin_memory,
        "persistent_workers": config.dataset.params.persistent_workers,
    }
    data_loader = DataLoader(train_dataset, **dataloader_params)
    ##################################
    #  Lr scheduler #
    #################################
    num_train_epochs = config.training.get('num_train_epochs', 1)
    config.training.max_train_steps = len(data_loader)*config.training.num_epoch
    
    min_lr_scale = float(config.lr_scheduler.params.get('min_lr', 0) ) /  optimizer_config.learning_rate
    if accelerate.__version__ < '1.0.0':
        lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=optimizer,
            num_training_steps=config.training.max_train_steps,
            num_warmup_steps=int(config.lr_scheduler.params.warmup_ratio * config.training.max_train_steps), 
            min_scale = min_lr_scale
        )
    else:
        lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes,
            num_warmup_steps=int(config.lr_scheduler.params.warmup_ratio * config.training.max_train_steps) * accelerator.num_processes,
            min_scale = min_lr_scale
        )
    
    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"] == 3:
        ref_model = accelerator.prepare(ref_model)
    

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
    eval_batch_size = config.experiment.get('eval_batch_size', 8)
    eval_text_len = config.experiment.get('eval_text_len', 128)

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_t2i}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Start training global step = {global_step}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
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

    global_step = 0
    for epoch in range(0, config.training.num_epoch):
        model.train()
        epoch_iterator = tqdm(data_loader, desc=f"Epoch {epoch}", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            batch_prompt_list = batch['batch_prompt_list']
            batch_chosen_image = batch['batch_chosen_image'].to(device)
            batch_rejected_image = batch['batch_rejected_image'].to(device)
            ( chosen_input_ids, 
             chosen_attention_mask_, 
             chosen_labels, 
             chosen_mask_prob, 
             chosen_image_tokens) = prepare_inputs_and_labels(
                batch_chosen_image,
                batch_prompt_list, 
                use_gen_projector=use_gen_projector, 
                t2i_gen_mode=t2i_gen_mode)
                

            ( rejected_input_ids, 
             rejected_attention_mask_, 
             rejected_labels, 
             rejected_mask_prob, 
             rejected_image_tokens ) = prepare_inputs_and_labels(
                batch_rejected_image, 
                batch_prompt_list, 
                use_gen_projector=use_gen_projector, 
                t2i_gen_mode=t2i_gen_mode)

            if use_causal_mask:
                chosen_attention_mask = chosen_attention_mask_
                rejected_attention_mask = rejected_attention_mask_
            else:
                chosen_attention_mask = create_attention_mask_predict_next(chosen_input_ids,
                                                                       pad_id=int(
                                                                           uni_prompting.sptids_dict['<|pad|>']),
                                                                       soi_id=int(
                                                                           uni_prompting.sptids_dict['<|soi|>']),
                                                                       eoi_id=int(
                                                                           uni_prompting.sptids_dict['<|eoi|>']),
                                                                       rm_pad_in_image=True,
                                                                       return_inverse_mask=True).to(mask_dtype)
            rejected_attention_mask = create_attention_mask_predict_next(rejected_input_ids,
                                                                         pad_id=int(
                                                                             uni_prompting.sptids_dict['<|pad|>']),
                                                                         soi_id=int(
                                                                             uni_prompting.sptids_dict['<|soi|>']),
                                                                         eoi_id=int(
                                                                             uni_prompting.sptids_dict['<|eoi|>']),
                                                                         rm_pad_in_image=True,
                                                                         return_inverse_mask=True).to(mask_dtype)

            def concatenated_forward(model: nn.Module, chosen_input_ids, rejected_input_ids, chosen_labels,
                                     rejected_labels, chosen_attention_mask, rejected_attention_mask, use_gen_projector):
                len_chosen = chosen_input_ids.shape[0]
                concatenated_input_ids = torch.cat((chosen_input_ids, rejected_input_ids), dim=0)
                new_labels = torch.cat((chosen_labels, rejected_labels), dim=0)
                if use_gen_projector:
                    soi_pos = -config.model.unigen.num_vq_tokens - 2 if concatenated_input_ids[0,-1] == uni_prompting.eos_token_id else  - config.model.unigen.num_vq_tokens - 1 
                    if hasattr(model, 'module'):
                        input_part1 =  model.module.llm.model.embed_tokens(concatenated_input_ids[:, :soi_pos])
                        input_image = model.module.get_gen_embed(concatenated_input_ids[:, soi_pos: soi_pos + config.model.unigen.num_vq_tokens].contiguous())
                        input_part2 = model.module.llm.model.embed_tokens(concatenated_input_ids[:, soi_pos + config.model.unigen.num_vq_tokens: ])
                        input_embeddings = torch.cat([input_part1, input_image, input_part2], 1)
                    else:
                        input_part1 =  model.llm.model.embed_tokens(concatenated_input_ids[:, :soi_pos])
                        input_image = model.get_gen_embed(concatenated_input_ids[:, soi_pos : soi_pos + config.model.unigen.num_vq_tokens])
                        input_part2 = model.llm.model.embed_tokens(concatenated_input_ids[:, soi_pos + config.model.unigen.num_vq_tokens: ])
                        input_embeddings = torch.cat([input_part1, input_image, input_part2], 1)
                else:
                    input_embeddings = None
                concatenated_attention_mask = torch.cat((chosen_attention_mask, rejected_attention_mask), dim=0)
                concatenated_attention_mask = concatenated_attention_mask.to(torch.bfloat16) if model.training else concatenated_attention_mask.to(torch.float32)
                all_logits = model(
                    input_ids=concatenated_input_ids,
                    input_embeddings=input_embeddings,
                    attention_mask=concatenated_attention_mask,
                    label_smoothing=config.training.label_smoothing,
                    t2i_mode=t2i_gen_mode,
                    batch_size_t2i=concatenated_input_ids.shape[0],
                )
                all_logits = all_logits.to(torch.float32)
                all_logps = get_batch_logps(
                    all_logits,
                    new_labels,
                    label_pad_token_id=-100,
                    num_vq_tokens=config.model.unigen.num_vq_tokens,
                    t2i_gen_mode=t2i_gen_mode
                )

                chosen_logps = all_logps[:len_chosen]
                rejected_logps = all_logps[len_chosen:]

                chosen_logits = all_logits[:len_chosen]
                rejected_logits = all_logits[len_chosen:]

                chosen_labels = new_labels[:len_chosen]
                rejected_labels = new_labels[len_chosen:]
                return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels)

            with accelerator.accumulate(model):
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    policy_chosen_logits,
                    policy_rejected_logits,
                    chosen_labels,
                    rejected_labels,
                ) = concatenated_forward(model, chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels,
                                         chosen_attention_mask, rejected_attention_mask, use_gen_projector)
                with torch.no_grad():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                    ) = concatenated_forward(
                        ref_model, chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels,
                        chosen_attention_mask, rejected_attention_mask, use_gen_projector
                    )[:2]

                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = reference_chosen_logps - reference_rejected_logps

                pi_logratios = pi_logratios.to(device)
                ref_logratios = ref_logratios.to(device)
                logits = pi_logratios - ref_logratios
                unscaled_dpo_losses = -F.logsigmoid(config.training.beta * logits)
                loss = config.training.dpo_coef*unscaled_dpo_losses.mean()

                avg_loss_t2i = accelerator.gather(loss.repeat(config.training.batch_size_t2i)).mean()
                avg_chosen_masking_rate = accelerator.gather(chosen_mask_prob.repeat(config.training.batch_size_t2i)).mean()
                avg_rejected_masking_rate = accelerator.gather(rejected_mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                should_log = (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                )
                if should_log:
                    logger_utils.log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if (global_step + 1) % config.experiment.log_every == 0:

                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "chosen_masking_rate": avg_chosen_masking_rate.item(),
                        "rejected_masking_rate": avg_rejected_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                global_step += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
      model = accelerator.unwrap_model(model)
      save_checkpoint(model, config, accelerator, global_step, save_last=True)

    accelerator.end_training()

if __name__ == "__main__":
    main()