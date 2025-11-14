#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from typing import Dict, Union, List
from pathlib import Path

from components.core import get_logger


THIRD_PARTY_PRETRAINED_CHECKPOINTS = {
    "magvitv2": "showlab/magvitv2",
    "siglip": "google/siglip-so400m-patch14-384/",
    "siglip2": "google/siglip2-so400m-patch14-384/",
    "siglip2_p16_naflex": "google/siglip2-so400m-patch16-naflex/",
    "qwen2.5-1_5b": "Qwen/Qwen2.5-1.5B-Instruct/",
    # https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt
    "vq_16": "llama_gen/",
    # https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
    "mask2former": "mask2former/",
}

logger = get_logger()


def all_checkpoints() -> Dict[str, str]:
    return {
        **THIRD_PARTY_PRETRAINED_CHECKPOINTS
    }


def real_checkpoint(ckpt_path_or_name: str, local_root_dir: str) -> str:
    a_path = Path(ckpt_path_or_name)
    if a_path.exists():
        return ckpt_path_or_name

    base_name = a_path.name
    available_checkpoints = {}
    for alias, ckpt_remote_path in all_checkpoints().items():
        available_checkpoints[Path(ckpt_remote_path).name] = ckpt_remote_path

    if base_name not in available_checkpoints:
        logger.info(f"Unknown checkpoint name: {base_name}.")
        return ckpt_path_or_name

    remote_path = available_checkpoints[base_name]
    # search local_path
    if local_root_dir:
      local_root_dir = Path(local_root_dir)
      ckpt_local_dir_or_file = local_root_dir / base_name
      return ckpt_local_dir_or_file.as_posix()
    else:
      return remote_path
    

def translate_checkpoint_names(checkpoint_list: List[str]) -> List[str]:
    registered_checkpoints = all_checkpoints()
    available_checkpoints = {}
    for alias, ckpt_remote_path in registered_checkpoints.items():
        available_checkpoints[Path(ckpt_remote_path).name] = alias

    checkpoints_norm = []
    for c in checkpoint_list:
        if c in available_checkpoints:
            checkpoints_norm.append(available_checkpoints[c])
        elif c in registered_checkpoints:
            checkpoints_norm.append(c)
        else:
            print(f"WARNING >>> Checkpoint {c} not recognized, dropping it.")

    return checkpoints_norm