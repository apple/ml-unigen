#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import json
import argparse
from tqdm import tqdm
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

def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '').replace("This image displays:", '').replace("This image displays ", '').replace(
        "The image shows ", '').replace("The image displays ", '').replace("The image appears to be ", '')
    return caption

class CustomDataset(Dataset):
    def __init__(self, metadata_path):
      with open(metadata_path) as fp:
        self.meta_data = [json.loads(line) for line in fp]
      self.data = []
      for item in self.meta_data:
        question = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n Now you need to convert an image description into fine-grained, related visual questions. The questions should comprehensively cover detailed visual facts of entities, attributes (e.g., color, count, texture, shape, and size), and relationships (e.g., spatial and non-spatial) between the entities mentioned in the description. Please complete the task by analyzing each clause in the sentence step by step. For each clause, first raise questions about whether each mentioned entity exists in the image. Then, raise questions about whether the attributes or relationships of the entities are accurately represented in the image. For an image accurately aligned with the description, all questions should be answered with yes; otherwise, they should be answered with n. "
                "Make sure all questions are able to be responded with yes or no and are connected with semicolon. Here are examples:"
                "Example 1: three black keys, four chickens and a fabric blanket. \n"
                "output: Are there keys?; Are there three keys?; Are the keys black?; Are there chickens?; Are there four chickens?; Is there a blanket?; Is the blanket fabric? \n"
                "Example 2: A person in a blue shirt and red and black apron is using a power tool, likely a drill, to assemble a white cabinet or shelving unit indoors. The floor is covered with light-colored wood or laminate material. \n"
                "output: Is there a person?; Is the person wearing a shirt; Is the shirt blue?; Is the person wearing a apron?; Is the apron red and black?; Is the person using a drill?; Is there a white cabinet or shelving unit?; Is the person using the drill indoors?; Is there light-colored wood on the floor?; Is there laminate material on the floor? \n"
                "Example 3: a large Ferris wheel with a digital clock showing the time as 11:00. The Ferris wheel is located in an urban area, as indicated by the modern buildings in the background. There is also a tree on the left side of the image, partially obscuring the view of the Ferris wheel. The sky appears clear, suggesting a sunny day. \n"
                "output: Is there a Ferris wheel?; Is there a digital clock?; Is the digital clock on the Ferris wheel?; Is the digital clock showing the time as 11:00?; Is the Ferris wheel located in an urban area?; Are there modern buildings in the background?; Is there a tree on the left side?; Is the sky clear and sunny? \n"  
                f"Please convert this image description: {remove_prefix(item['prompt'])} into fine-grained related visual questions. \n <|im_end|>\n"
                "<|im_start|>assistant\n")
        self.data.append(dict(id=item['id'], prompt=item['prompt'], question=question))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main(data_args):
    print(f"data_args: {data_args}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_args.gpu_id)
    os.makedirs(data_args.out_path, exist_ok=True)
    device = 0
    torch.cuda.set_device(device)
    
    dataset = CustomDataset(metadata_path=data_args.metadata_path)
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
        max_model_len=8192,
        max_num_seqs=1,
        device='cuda:0')

    
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=data_args.max_tokens,
                                     stop_token_ids=stop_token_ids)
    meta = []
    for data in tqdm(loader):
        item_id= data["id"][0]
        prompt=  data["prompt"][0]
        inputs = [{"prompt": data["question"][0]}]
        outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        questions = outputs[0].outputs[0].text.strip()
        for q in questions.split(';'):
            items = [
                {
                    "id": item_id,
                    "prompt": prompt,
                    "question": q,
                }
            ]
            meta.extend(items)
      
    save_path = os.path.join(
        data_args.out_path, 
        os.path.basename(data_args.metadata_path).split(".")[0] + "_question.json")

    with open(save_path, 'w', encoding='utf-8') as f:
      for item in meta:
          json_line = json.dumps(item, ensure_ascii=False)
          f.write(json_line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)
    
