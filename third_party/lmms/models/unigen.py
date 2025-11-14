#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/v0.3.0/lmms_eval/models/llava.py
# Copyright (c) 2025 LMMs-Lab
# licensed under MIT License

from typing import List, Optional, Tuple, Union
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval import utils
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent.parent))

from tqdm import tqdm
from training.prompting_utils import UniversalPromptingQwen2, create_attention_mask_for_mmu_vit
from data.transform import image_transform
from models.multimodal_encoder.builder import get_vision_tower
from models import UniGen
from omegaconf import OmegaConf
from loguru import logger as eval_logger
from data.llava.conversation import conv_templates
from data.llava.llava_data_vq_unified import DEFAULT_IMAGE_TOKEN
from transformers import AutoTokenizer

# from accelerate.state import AcceleratorState
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from accelerate import Accelerator, DistributedType
from datetime import timedelta
from pathlib import Path
from einops import rearrange
from PIL import Image
import glob
import re

def concat_images(images):
    num_images = len(images)
    if num_images == 0:
        raise ValueError("the images list should not be empty")

    widths, heights = zip(*(img.size for img in images))
    base_width = widths[0]
    base_height = heights[0]
    images = [img.resize((base_width, base_height)) for img in images]
    if num_images in [2, 3, 5]:
        total_width = base_width * num_images
        result = Image.new('RGB', (total_width, base_height))
        for i, img in enumerate(images):
            result.paste(img, (i * base_width, 0))
        return result
    elif num_images in [4, 6]:
        cols = 2 if num_images == 4 else 3
        rows = num_images // cols
        result = Image.new('RGB', (cols * base_width, rows * base_height))
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            result.paste(img, (col * base_width, row * base_height))
        return result
    else:
        raise ValueError(f"Concatation {num_images} images is not supported ")
    
@register_model("unigen")
class unigen(lmms):
    def __init__(
        self,
        config,
        pretrained: str=None,
        device: str = "cuda",
        device_map: str = "cuda:0",
        resolution: int=None,
        use_cache=True,
        batch_size: int=1,
        mm_input_mode='first',
        ckpt_base_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        # Setup accelerator.
        # assert batch_size == 1, f"Batch size should be 1 for Show-o, but got {batch_size}."
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load model.
        config = OmegaConf.load(config)
        try:
            self.resolution = config.dataset.preprocessing.get('mmu_resolution', config.dataset.preprocessing.get('resolution', resolution))
        except:
            self.resolution = resolution
        self.n_grid =  self.resolution // config.model.vision_tower.get('resolution', self.resolution)
        print(f"resolution={self.resolution}, n_grid={self.n_grid}")
        use_safetensors =  config.model.get('load_with_safetensors', None)
        pretrained_model_path = pretrained if pretrained is not None else config.model.unigen.pretrained_model_path
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
        print(f"load with local model: {pretrained_model_path}, use_safetensors: {use_safetensors}")

        if ckpt_base_path:
            config.model.vq_model.vq_model_name = os.path.join(ckpt_base_path, config.model.vq_model.vq_model_name)
            config.model.unigen.llm_model_path = os.path.join(ckpt_base_path, config.model.unigen.llm_model_path)

        assert config.model.unigen.w_und_encoder
        if ckpt_base_path:
            config.model.vision_tower.name = os.path.join(ckpt_base_path, config.model.vision_tower.name)
        self._vision_tower = get_vision_tower(config.model.vision_tower.name).to(accelerator.device)
        self.image_processor = self.vision_tower.image_processor

        self._model, msg = UniGen.from_pretrained(
            pretrained_model_path, use_safetensors=use_safetensors, ckpt_base_path=ckpt_base_path,
            output_loading_info=True
        )
        if self._model.config.vision_tower_name is None:
            self._vision_tower = get_vision_tower(config.model.vision_tower.name).to(accelerator.device)
            self.image_processor = self.vision_tower.image_processor
            if hasattr(self.image_processor, 'max_num_patches'):
                self.max_num_patches = config.model.vision_tower.max_num_patches
                self.image_processor.max_num_patches = self.max_num_patches
        else:
            self.image_processor = self._model.vision_tower.image_processor
            if hasattr(self.image_processor, 'max_num_patches'):
                self.max_num_patches = config.model.vision_tower.max_num_patches
                self.image_processor.max_num_patches = self.max_num_patches

        self._model.eval().cuda()
        print(f"loading model from {pretrained_model_path}: {msg}")

        model_version = "qwen_2.5"
        model_max_length=config.model.unigen.get("model_max_length", 32768) # follow LLaVA-Onevision
        tokenizer = AutoTokenizer.from_pretrained(config.model.unigen.llm_model_path, model_max_length=model_max_length, padding_side="right")
        self.uni_prompting = UniversalPromptingQwen2(tokenizer, 
                                    special_tokens=(
                                        "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                        "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                    ),
                                    max_seq_len=model_max_length,
                                    enable_reuse_tk=config.model.get("enable_reuse_tk", False), # reuse similar tokens in Qwen2 template, e.g., <|vision_start|>, <|vision_end|>
                                    task_token_first=config.model.get("task_token_first", True),
                                    ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
                                        
        config.model.unigen.vocab_size = len(self.uni_prompting.text_tokenizer) + config.model.unigen.codebook_size + 1
        config.model.unigen.llm_vocab_size = self.uni_prompting.text_tokenizer.vocab_size
        config.model.unigen.num_new_special_tokens = len(self.uni_prompting.text_tokenizer) -  config.model.unigen.llm_vocab_size
        
        self.config = config
        print('special tokens : \n', self.uni_prompting.sptids_dict)
            
        self.conv_template = model_version
        self.mm_input_mode = mm_input_mode
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                model = accelerator.prepare(model)
            self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.process_index
            if self._rank == 0:
                print(f"use_safetensors: {use_safetensors}, resolution: {resolution}, pretrained: {pretrained_model_path}")
                print(f"config: {config}")
            self._world_size = self.accelerator.num_processes
            if accelerator.is_local_main_process:
                print(f"Using {accelerator.num_processes} devices with data parallelism")
            print(f"initialize accelerator on gpu rank:{self._rank} world_size: {self._world_size}")
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
            print(f"use_safetensors: {use_safetensors}, resolution: {resolution}, pretrained: {pretrained_model_path}")
            print(f"config: {config}")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu
    
    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def vision_tower(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "_vision_tower"):
            return self._vision_tower
        elif hasattr(self.model, 'module'):
            return self.model.module.vision_tower
        else:
            return self.model.vision_tower

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for UniGen")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.uni_prompting.text_tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        print(f"processing requests with len={len(requests)}")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)
            # We assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            image_tensor = []
            spatial_shapes = []
            pixel_attention_mask = []
            image_low_res = []
            if len(flattened_visuals) > 1 and hasattr(self.image_processor, 'max_num_patches') and self.mm_input_mode == 'concat':
                flattened_visuals = [concat_images(flattened_visuals)]
            #     self.image_processor.max_num_patches = int(self.max_num_patches / len(flattened_visuals))
            for visual in flattened_visuals:
                if hasattr(self.image_processor, 'max_num_patches'):
                    image_output = self.image_processor.preprocess(visual, return_tensors='pt')
                    image_tensor.append(image_output["pixel_values"][0].to(device=self.device))
                    pixel_attention_mask.append(image_output["pixel_attention_mask"][0])
                    spatial_shapes.append(image_output["spatial_shapes"][0])
                elif self.n_grid == 1:
                    image_tensor.append(self.image_processor(visual, return_tensors="pt")["pixel_values"][0])
                else:
                    image_tensor.append(image_transform(visual, resolution=self.resolution).to(device=self.device))
                    if self.image_processor.size and self.image_processor.size != self.resolution:
                        image_low_res.append(self.image_processor(visual, return_tensors="pt")["pixel_values"][0])
            image_tensor = torch.stack(image_tensor, 0) 
            if len(image_low_res) > 0:
                image_low_res =  torch.stack(image_low_res, 0).to(device=self.device)
            else:
                image_low_res = None
            if len(pixel_attention_mask) > 0:
                spatial_shapes =  torch.stack(spatial_shapes, 0).long().to(device=self.device)
                pixel_attention_mask =  torch.stack(pixel_attention_mask, 0).to(device=self.device)
            else:
                spatial_shapes = None
                pixel_attention_mask = None
            if self.mm_input_mode == 'first':
                image_tensor = image_tensor[0].unsqueeze(0) #FIXME: a quick sol for multi-image input when bs = 1
                if image_low_res is not None:
                    image_low_res = image_low_res[0].unsqueeze(0)
                if spatial_shapes is not None:
                    spatial_shapes = spatial_shapes[0].unsqueeze(0) 
                    pixel_attention_mask = pixel_attention_mask[0].unsqueeze(0) 
            input_ids_mmu = []
            for i in range(len(batched_contexts)):
                context = batched_contexts[i].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                context = DEFAULT_IMAGE_TOKEN + '\n' + context
                context = context.strip()
                # Customized operation, get rid of <image> special token. Edited by Zechen
                context = context.replace(DEFAULT_IMAGE_TOKEN, "")
                prompts_input = context.strip()
                
                SYSTEM_PROMPT_LEN = 11
                conv = conv_templates[self.conv_template].copy()
                SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                conv.system = ''
                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input = []
                question_input.append(prompt_question.strip())

                input_ids = [self.uni_prompting.text_tokenizer(prompt, return_tensors="pt", padding="longest").input_ids  for prompt in question_input]
                input_ids = torch.stack(input_ids)
                
                input_ids = torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True, padding_value=self.uni_prompting.text_tokenizer.pad_token_id
                )
                input_ids = torch.tensor(input_ids).to(self.device).squeeze(0)
                input_ids_mmu.append(input_ids)
            input_ids_mmu = torch.cat(input_ids_mmu, 0)
            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 100
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 1
            gen_kwargs["top_p"] = int( gen_kwargs["top_p"])
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if len(res) == 0: # the first batch_idx
                print('gen_kwargs: ', gen_kwargs)
            
            bs_= image_tensor.shape[0]
            n_img = bs_// self.batch_size
            assert self.config.model.unigen.w_und_encoder
            input_ids_system = [self.uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids for _ in range(self.batch_size)]
            input_ids_system = torch.stack(input_ids_system, dim=0)
            assert input_ids_system.shape[-1] == SYSTEM_PROMPT_LEN
            input_ids_system = input_ids_system.to(self.device)
            input_ids_system = input_ids_system[0]
            
            if spatial_shapes is None:
                if self.n_grid > 1:
                    image_tensor = rearrange(image_tensor, "b c (n1 h ) (n2 w) -> (b n1 n2) c h w", h=self.config.model.vision_tower.resolution, w =self.config.model.vision_tower.resolution)
                    image_tensor = torch.cat([image_low_res, image_tensor], 0)
                images_feat = self.vision_tower(image_tensor)
                if self.n_grid > 1:
                    images_feat_low_res = images_feat[:bs_]
                    n_token = images_feat.shape[1]
                    images_feat = rearrange(images_feat[bs_: ], "(b n1 n2) (h w) c -> b (n1 h n2 w) c", n1=self.n_grid, n2=self.n_grid, h=int(n_token**0.5))
                    images_feat = torch.cat([images_feat_low_res, images_feat], 1)
                
                if hasattr(self.model, 'module'):
                    images_embeddings = self.model.module.mm_projector(images_feat) #B*n , N, C
                else:
                    images_embeddings = self.model.mm_projector(images_feat) #B*n , N, C
                if n_img == 1:
                    input_ids_part1, input_ids_part2, attention_mask, _ = self.uni_prompting((images_feat, input_ids_mmu, None, input_ids_system), 'mmu_conv')
                elif self.mm_input_mode == 'concat':
                    # TODO: option 1 simplying concating images w/o. interpolarion / resizing
                    images_embeddings = rearrange(images_embeddings, "(b n) m c -> b (n m) c", b=self.batch_size)
                    input_ids_part1, input_ids_part2, attention_mask, _ = self.uni_prompting((images_feat, input_ids_mmu, None, input_ids_system), 'mmu_conv')
                else:
                    raise NotImplementedError("do not support the multi-image input mode of " + self.mm_input_mode)
                if hasattr(self.model, 'module'):
                    part1 = self.model.module.llm.model.embed_tokens(input_ids_part1)
                    part2 = self.model.module.llm.model.embed_tokens(input_ids_part2)
                else:
                    part1 = self.model.llm.model.embed_tokens(input_ids_part1)
                    part2 = self.model.llm.model.embed_tokens(input_ids_part2)
                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
                attention_mask_llava = create_attention_mask_for_mmu_vit(input_embeddings, prefix_length=input_ids_part1.shape[1], num_tokens=self.config.model.vision_tower.num_tokens, num_images=n_img)
            else:
                images_feat = self.vision_tower(dict(pixel_values=image_tensor, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes))
                input_embeddings, _, _, input_ids_part1 = self.model.prepare_inputs_for_mmu(images_feat, spatial_shapes, input_ids_mmu, None, self.uni_prompting, input_ids_system)
                attention_mask_llava = create_attention_mask_for_mmu_vit(input_embeddings, prefix_length=input_ids_part1.shape[1], num_tokens=spatial_shapes, num_images=1)
                
            if self.config.model.get('use_causal_mask', False) or self.config.model.get('use_causal_mask_mmu', False):
                    cont_toks_list = self.model.generate(
                        input_embeddings=input_embeddings,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        pad_token_id=self.uni_prompting.text_tokenizer.eos_token_id,
                    )
            else:
                cont_toks_list = self.model.mmu_generate(input_embeddings=input_embeddings,
                        attention_mask=attention_mask_llava[0].unsqueeze(0),
                        max_new_tokens=gen_kwargs["max_new_tokens"], # self.config.inference.max_new_tokens
                        temperature=gen_kwargs["temperature"], #self.config.inference.temperature
                        top_k=gen_kwargs["top_p"], #int(self.config.inference.top_k)
                        eot_token=self.uni_prompting.text_tokenizer.eos_token_id
                )
                cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
            #FIXME: concat the answer
            text_outputs = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
            text_outputs = [out.strip() for out in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVA")
