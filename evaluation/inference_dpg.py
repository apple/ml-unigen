#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import sys
import json
import glob
import argparse
from pathlib import Path

current_file_path = Path(__file__).resolve()
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.insert(0, str(current_file_path.parent.parent.parent))

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from accelerate import PartialState
from transformers import AutoTokenizer
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything

from models import get_mask_chedule
from training.prompting_utils import UniversalPromptingQwen2, create_attention_mask_predict_next
from models.model_registry import get_model_creator
from utils.configuration import initialize_config

torch.set_grad_enabled(False)


def load_vision_encoders(config, device):
    vq_model = get_model_creator(config.model.vq_model.type).from_pretrained(
        config.model.vq_model.vq_model_name
    ).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    return vq_model


def load_uni_prompting(tokenizer, model_version, max_len, config):
    max_len_mode = config.model.get("max_len_mode", 'text')
    uni_prompting = UniversalPromptingQwen2(
        tokenizer,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        max_seq_len=config.dataset.preprocessing.max_seq_length + config.model.unigen.num_vq_tokens + 3 if max_len_mode == 'text' else max_len,
        # computing the maximum length of full sequence
        enable_reuse_tk=config.model.get("enable_reuse_tk", False),
        # reuse similar tokens in Qwen2 template, e.g., <|vision_start|>, <|vision_end|>
        # enable <|im_start|>, <|im_end|>
        task_token_first=config.model.get("task_token_first", True),
        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob
    )
    return uni_prompting

def load_unigen(config, device):
    use_safetensors = config.model.get('load_with_safetensors', None)
    ckpt_base_path = config.model.get("local_checkpoints", "")
    pretrained_model_path =os.path.join(ckpt_base_path, config.model.unigen.pretrained_model_path)

    print(f"load with local model: {pretrained_model_path}, use_safetensors: {use_safetensors}")
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

    # map the substrings to the version labels
    # (version, max_len, padding_side)
    version_map = {
        "Qwen2.5": ("qwen_2.5", 32_768, "right"),
    }
    # pick the first matching version, or None if no key is found
    model_version, max_len, padding_side = next(
        (version for substr, version in version_map.items() if substr in config.model.unigen.llm_model_path),
        None
    )
    max_len = config.model.unigen.get("model_max_length", max_len)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.unigen.llm_model_path,
        model_max_length=max_len,
        padding_side=padding_side
    )

    uni_prompting = load_uni_prompting(tokenizer, model_version, max_len, config)
    if "qwen" in model_version:
        config.model.unigen.vocab_size = len(uni_prompting.text_tokenizer) + config.model.unigen.codebook_size + 1
        config.model.unigen.llm_vocab_size = uni_prompting.text_tokenizer.vocab_size
        config.model.unigen.num_new_special_tokens = len(
            uni_prompting.text_tokenizer) - config.model.unigen.llm_vocab_size

    model = get_model_creator('unigen').from_pretrained(
        pretrained_model_path, use_safetensors=use_safetensors, ckpt_base_path=ckpt_base_path,
    ).to(device)
    model.eval()

    return model, uni_prompting, model_version


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
      

def dump_jsonl(data, f):
    lines = [json.dumps(x, ensure_ascii=False) for x in data]
    with open(f, "w", encoding="utf8") as fout:
        fout.write("\n".join(lines))


def get_batches(items, batch_size):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)
    return batches


def main():
    # Load prompts
    config = initialize_config()
    
    with open(config.dataset.validation_prompts_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    distributed_state = PartialState()
    device = distributed_state.device
    skip_grid  = config.inference.get('skip_grid', False)
    
    ckpt_base_path = config.model.get("local_checkpoints", "")
    if ckpt_base_path:
        config.model.vq_model.vq_model_name = os.path.join(ckpt_base_path, config.model.vq_model.vq_model_name)
        config.model.unigen.llm_model_path = os.path.join(ckpt_base_path, config.model.unigen.llm_model_path)

    if distributed_state.is_main_process:
      print(f"config: {config}")
      
    # load vision encoders
    vq_model = load_vision_encoders(config, device)

    # load uni-gen model
    model, uni_prompting, model_version = load_unigen(config, device)
    
    mask_token_id = model.config.mask_token_id
    metadata_len = len(metadatas)
    num_processes = distributed_state.num_processes
    seed_everything(config.inference.get('seed', 0))
    n_samples = config.inference.get('n_samples', 4)
    batch_size = config.inference.get("batch_size", n_samples)
    eval_text_len = config.model.get('eval_text_len', 128)
    t2i_gen_mode = config.model.get('t2i_gen_mode', 'mask')
    use_causal_mask  = config.model.get('use_causal_mask', False)
    assert use_causal_mask == True or t2i_gen_mode == 'mask'

    output_path_name = config.training.get('img_log_path',  f"dpg_bench_step{config.training.generation_timesteps}_scale{config.training.guidance_scale}")
    output_dir = os.path.join(config.experiment.output_dir, output_path_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # start generating images
    distributed_state.wait_for_everyone()
    for batch_index in trange(0, metadata_len, num_processes, desc="Generating images"):
        batch_raw = metadatas[batch_index: batch_index + num_processes]
        with distributed_state.split_between_processes(batch_raw) as batch:
            repeat_n = n_samples // batch_size
            index = batch_index + distributed_state.process_index
            all_samples = list()
            
            if batch is not None and index < metadata_len:
                print(f"Prompt ({index: >3}/{len(metadatas)}): '{batch[0]['text']}'")
                for idx in range(repeat_n):
                    prompt = [batch[0]['text']] * batch_size
                    image_tokens = torch.ones((len(prompt), config.model.unigen.num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
                    input_ids, attention_mask = uni_prompting((prompt, image_tokens, eval_text_len), 't2i_gen')

                    if config.training.guidance_scale > 0:
                        uncond_input_ids, attention_mask_ = uni_prompting(([''] * len(prompt), image_tokens, eval_text_len), 't2i_gen')
                        if use_causal_mask:
                            attention_mask = torch.cat([attention_mask, attention_mask_], dim=0)
                        else:
                            attention_mask = create_attention_mask_predict_next(
                                torch.cat([input_ids, uncond_input_ids], dim=0),
                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                rm_pad_in_image=True
                            )
                    else:
                        if use_causal_mask:
                            attention_mask = attention_mask
                        else:
                            attention_mask = create_attention_mask_predict_next(
                                input_ids,
                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                rm_pad_in_image=True
                            )
                        uncond_input_ids = None
            
                    if config.get("mask_schedule", None) is not None:
                        schedule = config.mask_schedule.schedule
                        args = config.mask_schedule.get("params", {})
                        mask_schedule = get_mask_chedule(schedule, **args)
                    else:
                        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

                    with torch.no_grad():
                        if t2i_gen_mode == 'mask':
                            gen_token_ids = model.t2i_generate(
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
                            gen_token_ids = model.t2i_generate_ar(
                                input_ids=input_ids,
                                uncond_input_ids=uncond_input_ids,
                                attention_mask=attention_mask,
                                guidance_scale=config.training.guidance_scale,
                                temperature=config.training.get("generation_temperature", 1.0),
                                text_vocab_size=len(uni_prompting.text_tokenizer),
                                image_token_num_per_image=config.model.unigen.num_vq_tokens,
                            )
                    
                    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.unigen.codebook_size - 1, min=0)
                    images = vq_model.decode_code(gen_token_ids)
                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                    if skip_grid:
                        images *= 255.0
                        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8) # b c h w -> b h w c
                        image = Image.fromarray(images[0])
                        image.save(os.path.join(output_dir, f"{batch[0]['item_id']}.png"))
                    else:
                        all_samples.append(images.cpu().clone())
            
                if not skip_grid and len(all_samples) > 0:
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=int(n_samples ** 0.5))
                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))
                    grid.save(os.path.join(output_dir, f"{batch[0]['item_id']}.png"))
                    del grid
                    del all_samples
            distributed_state.wait_for_everyone()
        
    print("Done.")

if __name__ == "__main__":
    main()