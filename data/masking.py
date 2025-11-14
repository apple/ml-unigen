#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/48966377380050e9a6ddb71df999ea2400391f9f/training/utils.py#L77
# Copyright 2024 NUS Show Lab.
# Licensed under the Apache License, Version 2.0 (the "License");

import math
import random
import torch

def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, is_train=True):

    def get_loss_weight(t, mask, min_val=0.3):
        return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]

    batch_size, seq_len = image_tokens.shape

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=image_tokens.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
            batch_idx,
            start_idx_height: start_idx_height + num_token_masked_height,
            start_idx_width: start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=image_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
            config.training.get("predict_all_tokens", False)
            or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob
