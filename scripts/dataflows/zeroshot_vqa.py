#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import json
import argparse

from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from vllm import LLM, SamplingParams


def dump_jsonl(data, f):
    lines = [json.dumps(x, ensure_ascii=False) for x in data]
    with open(f, "w", encoding="utf8") as fout:
        fout.write("\n".join(lines))


class CustomDataset(Dataset):
    def __init__(self, metadata_path, image_root, n_sample=20):
       with open(metadata_path) as fp:
        self.meta_data = [json.loads(line) for line in fp]
        self.data = []
        cur_img_id = None
        cur_prompt = None
        for item in self.meta_data:
            if cur_img_id != item['id']:
                if cur_img_id is not None:
                    for i in range(n_sample):
                        img_path = os.path.join(image_root, f"{cur_img_id}_{i}.png")
                        if not os.path.exists(img_path):
                            continue
                        self.data.append(dict(id=cur_img_id, img_path=img_path, prompt=cur_prompt, question=question_list, conversations=conv_list))
                cur_img_id = item['id']
                cur_prompt = item['prompt']
                conv_list, question_list = [], []
            if item['question'].strip() != "":
                conv = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question'].strip()} Please answer yes or no without explanation.<|im_end|>\n<|im_start|>assistant\n"
                question_list.append(item['question'].strip())
                conv_list.append(conv)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def main(data_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_args.gpu_id)
    os.makedirs(data_args.out_path, exist_ok=True)
    os.makedirs(os.path.join(data_args.out_path, 'metadata'), exist_ok=True)
    device = 0
    torch.cuda.set_device(device)

    dataset = CustomDataset(metadata_path=data_args.metadata_path, image_root=data_args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )
    
    llm = LLM(
        model=data_args.model_name,
        max_model_len=4096,
        max_num_seqs=1,
        device='cuda:0'
    )

    stop_token_ids = None
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=data_args.max_tokens,
                                     stop_token_ids=stop_token_ids)
    meta = []
    for data in tqdm(loader):
        img_path = data["img_path"][0]
        item_id= data["id"][0]
        prompt=  data["prompt"][0]
        answer_list  = []
        try:
            for question in data["conversations"]:
                inputs = [
                    {"prompt": question[0],
                    "multi_modal_data": {
                            "image": Image.open(img_path).convert("RGB")
                        },
                    }]
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
                answer_list.append('yes' if 'yes' in outputs[0].outputs[0].text.lower() else 'no')
            
            items = [
                {
                    "id": item_id,
                    "filename": img_path.split('/')[-1],
                    "prompt": prompt,
                    "question": [question[0] for question in data["question"]],
                    "answer": answer_list,
                    "score": sum([ans == 'yes' for ans in answer_list]) / float(len(answer_list)),
                }
            ]
            # print(f"{items[0]}")
            meta.extend(items)
        except:
            continue
    save_path = os.path.join(
        data_args.out_path, 
        os.path.basename(data_args.metadata_path).split(".")[0] + "_vqa_result.jsonl")

    with open(save_path, 'w', encoding='utf-8') as f:
      for item in meta:
          json_line = json.dumps(item, ensure_ascii=False)
          f.write(json_line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    args = parser.parse_args()
    main(args)
