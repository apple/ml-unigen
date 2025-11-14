#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/training/data.py
# Copyright 2024 NUS Show Lab.
# licensed under Apache License, Version 2.0 (the "License");

import os
import re
import json
import math
import random
import itertools
from functools import partial
from dataclasses import dataclass
from typing import List, Optional, Union

from PIL import Image
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import yaml
import pandas as pd
from tqdm import tqdm
import webdataset as wds
from webdataset import WebLoader
from braceexpand import braceexpand

import torch
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

from data.llava.llava_data_vq_unified import preprocess
from data.llava.llava_data_vq_unified import collate_fn as collate_fn_llava
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset


class WebLoader_(WebLoader):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
    
    def __len__(self):
        if hasattr(self, 'num_batches'):
            return self.num_batches
        elif hasattr(self, 'dataset'):
            return self.dataset.num_batches
        else:
            return None
        
@dataclass
class WdsConstants:
    """Constants for WebDataset."""
    # Pipe string for webdataset.
    pipe_str: str = "pipe: aws s3 cp"


def parse_data_dir(data_dir: Union[str, List[str]]) -> List[str]:
    """Parse data directory.
    Args:
        data_dir (Union[str, List[str]]): Data directory.
            If ends with json, will read the paths from the json file.
            Else, will use braceexpand to expand the path.
        shuffle (bool, optional): Shuffle the shards. Defaults to False.
    Returns:
        List[str]: List of data directories.
    """
    constants = WdsConstants()
    # use json file to get the tar file list
    if isinstance(data_dir, str) and data_dir.endswith(".json"):
        with open(data_dir, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        return [
            f"{constants.pipe_str} {tar_file} -"
            for tar_file in metadata["tar_file_list"]
        ]
    # directly pass the tar files
    if not isinstance(data_dir, list):
        data_dir = [data_dir]
    shards = []
    for r in data_dir:
        if r.startswith("s3://"):
            r = f"{constants.pipe_str} {r} -"
            shards += list(braceexpand(r))
        else:
            shards += [r]
    return shards


def replace_person_token(t):
    # Used for CC12M
    person_token = ["a person", "someone", "somebody"]
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files,  handler=handler)
    return samples


def image_transform(sample, enable_ti2i=False, resolution=256, visual_processor=None):
    if visual_processor:
        image = sample["images"]
        image_output = visual_processor.preprocess(image, return_tensors='pt')
        if hasattr(visual_processor, 'max_num_patches'):
            sample["images"] = image
            sample['images'] = image_output['pixel_values'][0]
            sample['pixel_attention_mask'] = image_output['pixel_attention_mask'][0]
            sample['spatial_shapes'] = image_output['spatial_shapes'][0]
        else:
            sample['images'] = image_output['pixel_values'][0]
    else:
        image = sample["images"]
        image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.CenterCrop((resolution, resolution))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
        sample["images"] = image
    return sample


def remove_prefix(caption):
    """Remove common prefixes from captions."""
    prefixes_to_remove = [
        'The image features ', 'The image presents ', "The image you've sent is, ",
        "In the center of the image, ", 'The image showcases ', 'The image is ',
        'The image captures ', "In the given image ", 'The image portrays ',
        "In the image, ", "In this image, we see ", 'The image depicts ',
        "This is ", "In this image, ", "This image captures ",
        "This image displays:", "This image displays ", 'The image shows ',
        'The image displays ', 'The image appears to be '
    ]

    for prefix in prefixes_to_remove:
        caption = caption.replace(prefix, '')
    return caption


class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            tokenizer: Optional[PreTrainedTokenizer],
            max_seq_length: int,
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            model_version='qwen_2.5',
            is_captioning: bool = False,
            add_caption_prompt: bool = False,
            short_caption_ratio= 0.5,
            data_dir: Optional[str] = '',
            caption_file="data/prompts/short_caption_prompt.json",
            visual_processor=None,            
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir('./configs'):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        self.short_caption_ratio = short_caption_ratio
        if self.add_caption_prompt:
            with open(caption_file, 'r') as f:
                self.caption_prompt = json.load(f)
                if model_version == 'qwen_2.5':
                    self.caption_prompt = [prompt +'<|im_end|>\n<|im_start|>assistant\n' for prompt in self.caption_prompt]
                else:
                    raise NotImplementedError(f"Do not support model {model_version}, please chose from qwen2.5")
        else:
            self.caption_prompt = None

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)
                input_ids = tokenizer(
                    text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                return input_ids[0]
            else:
                return text

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = [
                os.path.join(data_dir, x) if data_dir else x
                for x in itertools.chain.from_iterable(train_shards_path_or_url)
            ]
        elif data_dir:
            train_shards_path_or_url = os.path.join(data_dir, train_shards_path_or_url)

        img_processor = visual_processor if is_captioning else None
        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.map(self.process_caption, handler=wds.ignore_and_continue),
            wds.rename(
                images="jpg;png;jpeg;webp",
                input_ids="text;txt;caption",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(["images", "input_ids", "source"]))),
            wds.map(partial(image_transform, resolution=resolution, visual_processor=img_processor), handler=wds.warn_and_continue),
            wds.map_dict(
                input_ids=tokenize,
                handler=wds.warn_and_continue,
            ),
        ]
        train_shards_path_or_url = parse_data_dir(train_shards_path_or_url)
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        self.num_train_examples = num_train_examples
        self.num_batches = num_batches

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = WebLoader_(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            # num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            # persistent_workers=0,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def process_caption(self, sample):
        if 'txt' not in sample:
            sample['txt'] = ''
        sample['source'] = 'unknown'

        url_splits = sample['__url__'].lower().split('/')
        if len(url_splits) > 1:
            sample['source'] = url_splits[-2]
        
        if isinstance(sample['txt'], bytes):
            sample['txt'] = sample['txt'].decode('utf-8')

        url_lower = sample['__url__'].lower()
        if "json" in sample.keys() and isinstance(sample.get('json'), bytes):
            if sample.get('json').strip():
                try:
                    sample["json"] = json.loads(sample["json"].decode("utf-8"))
                except:
                    sample["json"] = dict(prompt=sample['txt'])

        if 'text2image' in url_lower:
            #TODO: we load original laion dataset w/o. re-captioned data
            # for captioning
            sample['source'] = 'text2image'
            sample['txt'] = sample['json']['prompt']
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + sample['txt'] 
            # for generation
            else:
                # randomly choose short and long captions
                if self.short_caption_ratio > 0 and random.random() < self.short_caption_ratio:
                    sample['txt'] = sample['txt'].split('.')[0]
                sample['txt'] = remove_prefix(sample['txt'])
            return sample
        elif 'journeydb' in url_lower:
            # Process JourneyDB samples.
            sample['source'] = 'journeydb'
            if self.short_caption_ratio > 0. and random.random() < self.short_caption_ratio:
                sample['txt'] = sample['txt'].split('.')[0]
            sample['txt'] = remove_prefix(sample['txt']).strip()
            return sample
        else:
            # Process default samples.
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + sample['txt']
            else:
                if self.short_caption_ratio > 0. and random.random() < self.short_caption_ratio:
                    sample['txt'] = sample['txt'].split('.')[0]
                sample['txt'] = replace_person_token(sample['txt'])
                sample['txt'] = remove_prefix(sample['txt']).strip()
            return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    def __len__(self):
        return self.num_train_examples


class ParquetLlavaTextOnlyDataset(Dataset):
    def __init__(
        self,
        parquet_dir_list,
        tokenizer,
        transform=None,
    ):
        self.parquet_dir_list = parquet_dir_list
        self.tokenizer = tokenizer
        self.transform = transform

        self.filepaths = []
        self.row_offsets = []
        total_rows = 0
        if isinstance(parquet_dir_list, str):
            parquet_dir_list = [parquet_dir_list]
        for parquet_dir in parquet_dir_list:
            for file in sorted(os.listdir(parquet_dir)):
                if file.endswith(".parquet"):
                    path = os.path.join(parquet_dir, file)
                    num_rows = pd.read_parquet(path, engine="pyarrow").shape[0]
                    self.filepaths.append(path)
                    self.row_offsets.append((total_rows, total_rows + num_rows))
                    total_rows += num_rows
        self.total_rows = total_rows

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(self.row_offsets):
            if start <= idx < end:
                try:
                    local_idx = idx - start
                    df = pd.read_parquet(self.filepaths[i], engine="pyarrow")
                    sample = df.iloc[local_idx].to_dict()
                    data_dict = preprocess([sample["conversations"]], self.tokenizer)
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0])
                    return data_dict
                except Exception as e:
                    print(f'Dataset iteration error at idx {idx}: {e}')
                
                
class ParquetTextDataset(Dataset):
    def __init__(
        self,
        parquet_dir_list,
        tokenizer,
        transform=None,
        max_length=8000,
        num_rows = 174919
    ):
        self.parquet_dir_list = parquet_dir_list
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        self.filepaths = []
        self.row_offsets = []
        total_rows = 0
        if isinstance(parquet_dir_list, str):
            parquet_dir_list = [parquet_dir_list]
        for parquet_dir in parquet_dir_list:
            for idx, file in enumerate(tqdm(sorted(os.listdir(parquet_dir)))):
                if file.endswith(".parquet"):
                    path = os.path.join(parquet_dir, file)
                    # num_rows = pd.read_parquet(path, engine="pyarrow").shape[0]
                    self.filepaths.append(path)
                    self.row_offsets.append((total_rows, total_rows + num_rows))
                    total_rows += num_rows
        self.total_rows = total_rows

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(self.row_offsets):
            if start <= idx < end:
                try:
                    local_idx = idx - start
                    df = pd.read_parquet(self.filepaths[i], engine="pyarrow")
                    sample = df.iloc[local_idx].to_dict()
                    text = sample['content']
                    if len(text) > self.max_length:
                        start_index = random.randint(0, len(text) - self.max_length - 1)
                        selected_text = text[start_index:start_index + self.max_length]
                    else:
                        selected_text = text
                    return {'input_ids': selected_text}
                except Exception as e:
                    print('Dataset iteration error at idx {idx}: {e}')

def make_pretrain_lm_dataloader(train_lm_shards_path_or_url,
    tokenizer,
    batch_size,
    num_workers,
    world_size,
    local_rank,
    max_length,
    repeat_n=1,
):
    if repeat_n > 1:
        if isinstance(train_lm_shards_path_or_url, str):
            train_lm_shards_path_or_url = [train_lm_shards_path_or_url] * repeat_n
    else:
        train_lm_shards_path_or_url = train_lm_shards_path_or_url * repeat_n
    train_dataset = ParquetTextDataset(train_lm_shards_path_or_url, tokenizer)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=default_collate,
        sampler=datasampler
    )
    return dataloader
    
def make_supervised_lm_dataloader(
    train_lm_shards_path_or_url,
    tokenizer,
    batch_size,
    num_workers,
    world_size,
    local_rank,
    max_length,
    repeat_n=1,
    add_system_prompt=False,
):
    if repeat_n > 1:
        if isinstance(train_lm_shards_path_or_url, str):
            train_lm_shards_path_or_url = [train_lm_shards_path_or_url] * repeat_n
        else:
            train_lm_shards_path_or_url = train_lm_shards_path_or_url * repeat_n
    
    train_dataset = ParquetLlavaTextOnlyDataset(train_lm_shards_path_or_url, tokenizer)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn_llava,
            tokenizer=tokenizer,
            add_system_prompt=add_system_prompt,
            max_length=max_length,
        ),
        sampler=datasampler
    )
    return dataloader
