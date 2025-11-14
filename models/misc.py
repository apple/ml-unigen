#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/models/misc.py
# Copyright 2024 NUS Show Lab.
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any, Optional, Union
# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
# Runtime type checking decorator
from typeguard import typechecked as typechecker

import torch
from omegaconf import OmegaConf, DictConfig

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def broadcast(tensor, src=0):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=src)
    return tensor


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    if '--local-rank' in cfg:
        del cfg['--local-rank']
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    return OmegaConf.structured(fields(**cfg))