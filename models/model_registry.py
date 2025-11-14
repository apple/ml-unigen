#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from models import UniGen, MAGVITv2, SigLipVisionTower
from typing import Dict


class ModelRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, key, value):
        self._registry[key.lower()] = value

    def get(self, key):
        key_lower = key.lower()
        # direct match
        if key_lower in self._registry:
            return self._registry[key_lower]
        # search for key
        for model_key in self._registry.keys():
            if model_key in key_lower:
                return self._registry[model_key]
        raise ValueError(
            f"Unsupported model type: {key}. Supported types: {list(self._registry.keys())}"
        )

    def update(self, defines: Dict):
        self._registry.update(defines)


def register_model_class(keyword):
    """Decorator to register a model class."""
    def decorator(cls):
        MODEL_REGISTRY.register(keyword.lower(), cls)
        return cls
    return decorator

def register_model_func(keyword):
    """Decorator to register a model through factory function. """
    def decorator(func):
        MODEL_REGISTRY.register(keyword.lower(), func)
        return func
    return decorator


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.update({
    'magvitv2': MAGVITv2,
    'siglip': SigLipVisionTower,
    'unigen': UniGen,
})


def get_model_creator(keyword):
    return MODEL_REGISTRY.get(keyword)


def model_from_name(name):
    creator_cls = MODEL_REGISTRY.get(name.lower())
    return creator_cls(name)


