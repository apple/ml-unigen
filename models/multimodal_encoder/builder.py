#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from .siglip_encoder import SigLipVisionTower
from .siglip2_encoder import SigLip2VisionTower

def get_vision_tower(model_name, freeze=True):
    if "siglip2"  in model_name and 'naflex' in model_name:
        return SigLip2VisionTower(model_name, freeze=freeze)
    elif "siglip"  in model_name :
        return SigLipVisionTower(model_name, freeze=freeze)
    else:
        raise ValueError(f"model_type {model_name} not supported.")
