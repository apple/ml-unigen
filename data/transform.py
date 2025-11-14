#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import math
import torch
from torchvision import transforms
from PIL import Image


def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image

def pad_image(image, target_resolution, value=0):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new('RGB', (target_width, target_height), (value, value, value))
    paste_x = (target_width - original_width) // 2
    paste_y = (target_height - original_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    return new_image
