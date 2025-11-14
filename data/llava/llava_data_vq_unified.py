#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/llava/llava_data_vq_unified.py
# Copyright 2024 NUS Show Lab.
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import copy
import json
from functools import partial

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from data.transform import image_transform
from data.llava import conversation as conversation_lib


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN="<soi>"
DEFAULT_IM_END_TOKEN="<eoi>"
IMAGE_TOKEN_INDEX=-200
IGNORE_INDEX = -100


def preprocess_multimodal(sources, image_token=DEFAULT_IMAGE_TOKEN):
    for source in sources:
        for sentence in source:
            if image_token in sentence['value']:
                sentence['value'] = sentence['value'].replace(image_token, '').strip()
                sentence['value'] = image_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(image_token, "")
                sentence['value'] = sentence['value'].strip()

            # Customized operation, get rid of <image> special token. Edited by Zechen
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
            sentence['value'] = sentence['value'].strip()
    return sources


def preprocess_qwen(
    sources, 
    tokenizer,
    system_message =  "You are a helpful assistant."):
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}
    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens_idx =  [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    input_ids_system = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]
        input_id, target = [], []
        # New version, use apply chat template
        # Build system message for each sentence
        input_ids_system.append(tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}]))

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        
                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        # TODO: stage 2 for processing muti-image
        # for idx, encode_id in enumerate(input_id):
        #     if encode_id in unmask_tokens_idx:
        #         target[idx] = encode_id
        #     if encode_id == image_token_index:
        #         input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    input_ids_system = torch.tensor(input_ids_system, dtype=torch.long)
        
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        input_ids_system=input_ids_system
    )


def preprocess_v0(
        sources,
        tokenizer,
        system_message =  "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."):
    # Let's assume has_image is false, since we will process the image token separately
    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [system_message for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )

def preprocess(sources,  tokenizer):
    if "phi" in tokenizer.name_or_path.lower():
        return preprocess_v0(sources, tokenizer)
    if "qwen" in tokenizer.name_or_path.lower():
        return preprocess_qwen(sources, tokenizer)

class LLaVADataset(Dataset):
    def __init__(self,
        tokenizer,
        data_file_path,
        image_root,
        resolution=256,
        diable_text_rich=False,
    ):
        super(LLaVADataset, self).__init__()

        self.tokenizer = tokenizer
        self.resolution = resolution
        data_file_path = data_file_path
        self.image_root = image_root
        if "phi" in tokenizer.name_or_path.lower():
            conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2.5"]
        
        
        if not isinstance(data_file_path, (tuple,list)):
            data_file_path = [data_file_path]
        self.list_data_dict = []
        for data_file in data_file_path:
            with open(data_file, 'r') as f:
                data = json.load(f)
            for item in data:
                if 'image' in item.keys():
                    #FIXME: a temporary solution for ShareGPT4V & LLaVA to remove text-riched dataset
                    if diable_text_rich and "ocr_vqa" not in item["image"] and "textvqa" not in item["image"]:
                        self.list_data_dict.append(item)
                    elif not diable_text_rich:
                        self.list_data_dict.append(item)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_root
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = image_transform(image, resolution=self.resolution)
        except:
            print(f"Read image error {image_file}. Use dummy data.")
            crop_size = self.resolution
            image = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        data_dict = preprocess(sources, self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.resolution
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


def collate_fn(
        instances,
        tokenizer=None,
        add_system_prompt=False,
        max_length=77,
):
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_INDEX
    )
    
    if not add_system_prompt:
        if input_ids.shape[-1] < max_length:
            offset = max_length - input_ids.shape[-1]
            pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, pad_tube], dim=1)

            offset = max_length - labels.shape[-1]
            pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
            labels = torch.cat([labels, pad_tube], dim=1)

        min_max_len = min(max_length, tokenizer.model_max_length)

        input_ids = input_ids[:, :min_max_len]
        labels = labels[:, :min_max_len]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
        elif 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
        else:
            images = None
            
        if images:
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
    else:
        input_ids_system = torch.stack(input_ids_system, dim=0)
        offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]
        if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
            pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, pad_tube], dim=1)
            pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
            labels = torch.cat([labels, pad_tube], dim=1)

        min_max_len = min(
            max_length - input_ids_system.shape[-1],
            tokenizer.model_max_length - input_ids_system.shape[-1],
        )

        input_ids = input_ids[:, :min_max_len]
        labels = labels[:, :min_max_len]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            input_ids_system=input_ids_system,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
        elif 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
        else:
            images = None
                
        if 'pixel_attention_mask' in instances[0]:
            pixel_attention_mask = [instance['pixel_attention_mask'] for instance in instances]
            batch['pixel_attention_mask'] = torch.stack(pixel_attention_mask)
        
        if 'spatial_shapes' in instances[0]:
            spatial_shapes = [instance['spatial_shapes'] for instance in instances]
            batch['spatial_shapes'] = torch.stack(spatial_shapes)
               
        if images:
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

    return batch


def get_instruct_data_loader(
    tokenizer,
    batch_size,
    num_workers,
    world_size,
    local_rank,
    max_length,
    data_path,
    image_root,
    resolution,
    disable_text_rich=False,
    add_system_prompt=False
):
    train_dataset = LLaVADataset(
        tokenizer,
        data_path,
        image_root,
        resolution,
        disable_text_rich
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            add_system_prompt=add_system_prompt,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader

