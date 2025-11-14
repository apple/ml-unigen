#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/training/prompting_utils.py
# Copyright 2024 NUS Show Lab.
# licensed under Apache License, Version 2.0 (the "License");

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2TokenizerFast
import copy 

class UniversalPromptingQwen2():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|think_start|>", "<|think_end|>"),
                 ignore_id=-100, max_seq_len=None, cond_dropout_prob=0.1, enable_reuse_tk=False, task_token_first=False):
        """
        :param text_tokenizer: original text tokenizer
        special_tokens={'eos_token': '<|im_end|>', 'pad_token': '[PAD]', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False),
        
        """
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        
        self.enable_reuse_tk = enable_reuse_tk
        self.task_token_first = task_token_first
        if enable_reuse_tk:
            self.sptids_dict = dict()
            special_tokens=list(special_tokens)
            if "<|soi|>" in special_tokens:
                special_tokens.remove('<|soi|>')
                self.sptids_dict['<|soi|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|vision_start|>'])) 
            if "<|eoi|>" in special_tokens:
                special_tokens.remove('<|eoi|>')
                self.sptids_dict['<|eoi|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|vision_end|>'])) 
            if "<|sov|>" in special_tokens:
                special_tokens.remove('<|sov|>')
                self.sptids_dict['<|sov|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|vision_start|>'])) 
            if "<|eov|>" in special_tokens:
                special_tokens.remove('<|eov|>')
                self.sptids_dict['<|eov|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|vision_end|>'])) 
            self.text_tokenizer.add_tokens(special_tokens)
            for token in special_tokens:
                self.sptids_dict[token] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token]))
        else:
            self.text_tokenizer.add_tokens(list(special_tokens))
            self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in special_tokens}
   
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        self.bos_token_id = self.text_tokenizer.convert_tokens_to_ids(['<|im_start|>'])[0]
        self.eos_token_id = self.text_tokenizer.eos_token_id
        self.sptids_dict['<|im_start|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|im_start|>']))
        self.sptids_dict['<|im_end|>'] = torch.tensor(self.text_tokenizer.convert_tokens_to_ids(['<|im_end|>'])) 
    
        self.max_seq_len = max_seq_len
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob
        
    def t2i_prompt(self, text_ids, image_ids, labels):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))

        if self.task_token_first:
            conv_start_ids = self.text_tokenizer("<|t2i|><|im_start|>user\n").input_ids
        else:
            conv_start_ids = self.text_tokenizer("<|im_start|><|t2i|>user\n").input_ids
        conv_end_ids = self.text_tokenizer("<|im_end|>\n<|im_start|>assistant\n").input_ids
        
        
        for i in range(len(text_ids)):
            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                text_ids[i] = []

            temp_ids = conv_start_ids + text_ids[i] + conv_end_ids

            if self.max_seq_len >= len(temp_ids) + image_ids.shape[1] + 2:
                temp_masks = [0] *(self.max_seq_len - len(temp_ids) - image_ids.shape[1] - 2) + [1] * (len(temp_ids) + image_ids.shape[1] + 2)
                temp_ids = [self.pad_id] * (self.max_seq_len - len(temp_ids) - image_ids.shape[1] - 2) + temp_ids # pad left
            else:
                temp_masks = [1] *  self.max_seq_len
                temp_ids = temp_ids[:self.max_seq_len - image_ids.shape[1] - 2]  

            # there's no need to add |im_end|
            temp_label_ids = torch.cat([
                # torch.tensor(temp_ids).to(device) * self.ignore_id, 
                torch.tensor([self.ignore_id] * len(temp_ids)).to(device) ,
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device),
                # self.sptids_dict['<|im_end|>'].to(device)
                ], dim=0)
        
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                # self.sptids_dict['<|im_end|>'].to(device)
                ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2i_gen_prompt(self, text_ids, image_ids, max_len=None):
        # to provide evaluation during training 
        # no condition drop fot t2i_gen
        # no returns of labels 
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        
        if self.task_token_first:
            conv_start_ids = self.text_tokenizer("<|t2i|><|im_start|>user\n").input_ids
        else:
            conv_start_ids = self.text_tokenizer("<|im_start|><|t2i|>user\n").input_ids
        conv_end_ids = self.text_tokenizer("<|im_end|>\n<|im_start|>assistant\n").input_ids
        
        if max_len is None:
            max_len = max([len(ids) for ids in text_ids]) + len(conv_start_ids) + len(conv_end_ids) + 2 + image_ids.shape[-1] 
        else:
            max_len = max_len +  len(conv_start_ids) + len(conv_end_ids) + 2 + image_ids.shape[-1]
        max_len = min(max_len, self.max_seq_len)
        
        for i in range(len(text_ids)):
            temp_ids = conv_start_ids + text_ids[i] + conv_end_ids
            if max_len >= len(temp_ids) + image_ids.shape[1] + 2:
                temp_masks = [0] *(max_len - len(temp_ids) - image_ids.shape[1] - 2) + [1] * (len(temp_ids) + image_ids.shape[-1] + 2)
                temp_ids = [self.pad_id] * (max_len - len(temp_ids) - image_ids.shape[1] - 2) + temp_ids # pad left
            else:
                temp_masks = [1] *max_len
                temp_ids = temp_ids[:max_len - image_ids.shape[1] - 2 - len(conv_end_ids)] + conv_end_ids
            
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
            ], dim=0)
            
            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def lm_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = []
            else:    # text_ids[0] == self.sot_token_id
                text_ids[i] =  text_ids[i]
            
            temp_ids =  [int(self.sptids_dict['<|im_start|>'])] + text_ids[i] +  [int(self.sptids_dict['<|im_end|>'])] 
                
            if max_seq_len >= len(temp_ids): # pad to right
                temp_labels_ids = temp_ids + [self.ignore_id] * (max_seq_len - len(temp_ids))
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
                temp_ids = temp_ids + [self.pad_id] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_labels_ids = torch.tensor(temp_labels_ids)
            temp_masks = torch.tensor(temp_masks)
            
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)
    
    def mmu_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []
            if len(text_ids[i]) == 0:
                text_ids[i] = []
            else:    # text_ids[0] == self.sot_token_id
                text_ids[i] = text_ids[i]
            
            temp_ids = text_ids[i]
            temp_ids = temp_ids
            if self.max_seq_len >= len(temp_ids) + image_ids.shape[1] + 5: # [mmu], [eoi], [soi] [im_start]
                temp_masks =  [1] * (len(temp_ids) + image_ids.shape[1] + 5) + [0] *(self.max_seq_len - len(temp_ids) - image_ids.shape[1] - 5) 
                temp_ids = temp_ids  + [int(self.sptids_dict['<|im_end|>'])] +  [self.pad_id] * (self.max_seq_len - len(temp_ids) - image_ids.shape[1] - 5) # pad right
            else:
                temp_masks = [1] * self.max_seq_len
                temp_ids =  temp_ids[:self.max_seq_len - image_ids.shape[1] - 5]  + [int(self.sptids_dict['<|im_end|>'])]
   
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device), 
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)
            if self.task_token_first:
                temp_ids = torch.cat([
                    self.sptids_dict['<|mmu|>'].to(device),  # task token
                    self.sptids_dict['<|im_start|>'].to(device),  # task token
                    self.sptids_dict['<|soi|>'].to(device),
                    image_ids[i],
                    self.sptids_dict['<|eoi|>'].to(device),
                    torch.tensor(temp_ids).to(device),
                    ], dim=0)
            else:
                temp_ids = torch.cat([
                    self.sptids_dict['<|im_start|>'].to(device),  # task token
                    self.sptids_dict['<|mmu|>'].to(device),  # task token
                    self.sptids_dict['<|soi|>'].to(device),
                    image_ids[i],
                    self.sptids_dict['<|eoi|>'].to(device),
                    torch.tensor(temp_ids).to(device),
                ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def mmu_conv(self, images_embeddings, input_ids, label_ids, input_ids_system):
        device = input_ids.device
        discrete_image_input = images_embeddings.ndim == 2
        img_seq_len = images_embeddings.shape[1]
        
        if label_ids is None:
            label_ids = copy.deepcopy(input_ids)
        if self.task_token_first:
            input_ids_part1 = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|im_start|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|soi|>']).to(device)], dim=1).long()
        else:
            input_ids_part1 = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|im_start|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|soi|>']).to(device)], dim=1).long()

        input_ids_part2 =  torch.cat([
        (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|eoi|>']).to( device),
        input_ids[:, 1:]], dim=1).long()
        
        if input_ids_system is not None:
            input_ids_part1 = torch.cat([input_ids_system, input_ids_part1], dim=1).long()
            label_ids = torch.cat([
                torch.ones_like(input_ids_system) * self.ignore_id,  # ignore system prompt
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], img_seq_len) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                label_ids[:, 1:].to(device)
            ], dim=1).long()
        else:
            label_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], img_seq_len) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                label_ids[:, 1:].to(device)
            ], dim=1).long()
        attention_mask = torch.zeros((input_ids.shape[0], self.max_seq_len), device=device).to(torch.bool)
        position_ids = torch.zeros((input_ids.shape[0], self.max_seq_len), dtype=torch.long, device=device)
        eoi_pos = torch.where(input_ids_part2.flip(-1) == self.eos_token_id)
        eoi_id, eoi_len = 0, len(eoi_pos[0])
        for i in range(input_ids_part2.shape[0]):
            if eoi_id < eoi_len and eoi_pos[0][eoi_id] == i:
                cur_len = input_ids_part2.shape[1] - eoi_pos[1][eoi_id] + input_ids_part1.shape[1] + img_seq_len
                while eoi_id + 1 < eoi_len and eoi_pos[0][eoi_id + 1 ] == i:
                    eoi_id +=1
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
                eoi_id +=1
            else:
                cur_len = input_ids_part2.shape[1]
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
        
        if discrete_image_input:
            input_ids = torch.cat([input_ids_part1, images_embeddings, input_ids_part2], dim=1).long()
            return input_ids, attention_mask, label_ids
        else:
            return input_ids_part1, input_ids_part2, attention_mask, label_ids
            
    def mmu_embed(self, image_ids, text_ids):
        
        device= image_ids.device
        if image_ids.ndim == 3:
            img_seq_len =[image_ids.shape[1]] * image_ids.shape[0]
        else:
            spatial_shapes = image_ids
            img_seq_len = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        prefix_ids = []
        attention_masks = []
        suffix_ids = []
        label_ids = []
        
        if self.task_token_first:
            conv_start_ids = self.text_tokenizer("<|mmu|><|im_start|>user\n<|soi|>").input_ids
        else:
            conv_start_ids = self.text_tokenizer("<|im_start|><|mmu|>user\n<|soi|>").input_ids
        
        conv_end_ids = self.text_tokenizer("<|im_end|>\n<|im_start|>assistant\n").input_ids
        
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []
            temp_ids =  [int(self.sptids_dict['<|eoi|>'])] + text_ids[i] 
            eos_pos = temp_ids.index(int(self.sptids_dict['<|im_end|>'])) + len(conv_end_ids)
            if self.max_seq_len >= len(temp_ids) + img_seq_len[i] + len(conv_start_ids) + 1:
                temp_masks = [1] * (len(temp_ids) + img_seq_len[i]  + len(conv_start_ids) + 1) +  [0] *(self.max_seq_len - len(temp_ids) - img_seq_len[i] - len(conv_start_ids) - 1)
                temp_ids = temp_ids  +  [int(self.sptids_dict['<|im_end|>'])] +  [self.pad_id] * (self.max_seq_len - len(temp_ids) - img_seq_len[i] - len(conv_start_ids) - 1) # pad right
            else:
                temp_masks = [1] * self.max_seq_len
                temp_ids = temp_ids[:self.max_seq_len - img_seq_len[i] -  len(conv_start_ids) ] 
                
            # prompting -- [task token] [soi] [image tokens] [eoi]         
            temp_label_ids = torch.cat([
                torch.ones(len(conv_start_ids)).to(device) * self.ignore_id,
                torch.ones(img_seq_len[i]).to(device) * self.ignore_id,
                torch.ones(eos_pos).to(device) * self.ignore_id,
                torch.tensor(temp_ids[eos_pos:]).to(device),
            ], dim=0).long()
            
            temp_ids = torch.cat([
                torch.ones(img_seq_len[i]).to(device) * self.pad_id,
                torch.tensor(temp_ids).to(device),], dim=0).long()
            
            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)
            prefix_temp_ids =  torch.tensor(conv_start_ids, device=device)
            temp_masks = torch.tensor(temp_masks).to(device)
            suffix_ids.append(temp_ids.unsqueeze(0))
            prefix_ids.append(prefix_temp_ids.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(prefix_ids, dim=0), torch.cat(suffix_ids, dim=0),torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)
    def lm_conv(self, input_ids):
        device = input_ids.device
        attention_mask = torch.zeros((input_ids.shape[0], self.max_seq_len), device=device).to(torch.bool)
        position_ids = torch.zeros((input_ids.shape[0], self.max_seq_len), dtype=torch.long, device=device)

        eoi_pos = torch.where(input_ids.flip(-1) == self.eos_token_id) # find the first pad token
        eoi_id, eoi_len = 0, len(eoi_pos[0])
        for i in range(input_ids.shape[0]):
            if eoi_id < eoi_len and eoi_pos[0][eoi_id] == i:
                cur_len = input_ids.shape[1] - eoi_pos[1][eoi_id] 
                while eoi_id + 1 < eoi_len and eoi_pos[0][eoi_id + 1 ] == i:
                    eoi_id +=1
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
                eoi_id +=1
            else:
                cur_len = input_ids.shape[1]
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
        
        return attention_mask, position_ids
    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            max_len = None if len(input) == 2 else input[2]
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids, image_ids, max_len)
            
        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])
        
        elif task == "lm_conv":
            sequence_ids_with_masks = self.lm_conv(input)

        elif task == "mmu":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])['input_ids']
            sequence_ids_with_masks = self.mmu_prompt(image_ids, text_ids)
        
        elif task == "mmu_conv":
            sequence_ids_with_masks = self.mmu_conv(input[0],  input[1], input[2], input[3])
        
        elif task == "mmu_emb":
            text_ids = self.text_tokenizer(input[1])['input_ids']
            sequence_ids_with_masks = self.mmu_embed(input[0], text_ids)
        else:
            raise NotImplementedError
        return sequence_ids_with_masks

class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                 max_text_len=8000, max_seq_len=None, ignore_id=-100, cond_dropout_prob=0.1):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_tokenizer.add_tokens(list(special_tokens))
        self.sptids_dict = {token: torch.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in
                            special_tokens}
        self.sptids_dict['<|sot|>'] = torch.tensor([self.text_tokenizer.bos_token_id])
        self.sptids_dict['<|eot|>'] = torch.tensor([self.text_tokenizer.eos_token_id])
        self.sptids_dict['<|pad|>'] = torch.tensor([self.text_tokenizer.pad_token_id])
        # plus 1 because at this time we add a task token before
        self.max_text_len = max_text_len + 1
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids('[PAD]')
        self.ignore_id = ignore_id
        self.eos_token_id =  self.sptids_dict['<|eot|>']
        self.cond_dropout_prob = cond_dropout_prob
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.text_tokenizer.model_max_length
    
    def t2i_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2i|>']), self.text_tokenizer.bos_token_id, self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids) + image_ids.shape[-1] + 3)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                # torch.tensor(temp_ids).to(device),
                torch.tensor([self.ignore_id] * len(temp_ids)).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2i_gen_prompt(self, text_ids, image_ids, max_len=None):

        device = image_ids.device
        sequence_ids = []
        
        if max_len is not None and max_len < 0:
            max_len = max([len(tmp_ids) for tmp_ids in text_ids])
            max_len += 3 
            
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if max_len is not None and len(temp_ids) <= max_len:
                temp_ids = [self.pad_id] * (max_len - len(temp_ids)) + temp_ids # pad left
            elif max_len is not None:
                temp_ids = temp_ids[:max_len- 1]  +  [self.text_tokenizer.eos_token_id]
            elif self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            sequence_ids.append(temp_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), None

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len):

        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.ignore_id] * (max_seq_len - len(temp_ids))
                temp_ids = temp_ids + [self.pad_id] * (max_seq_len - len(temp_ids))
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.tensor(temp_ids)
            temp_masks = torch.tensor(temp_masks)
            temp_labels_ids = torch.tensor(temp_labels_ids)

            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def mmu_prompt(self, image_ids, text_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.pad_id] * (max_text_len - len(temp_ids))
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * (max_text_len - len(temp_ids))
            else:
                # should add the eos token
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(image_ids[i]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                self.sptids_dict['<|mmu|>'].to(device),  # task token
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2v_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2v|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2v|>']), self.text_tokenizer.bos_token_id,
                            self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids) + image_ids.shape[-1] + 3)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|sov|>'].to(device),
                labels[i],
                self.sptids_dict['<|eov|>'].to(device)
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|sov|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eov|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def t2v_gen_prompt(self, text_ids, image_ids):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2v|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * len(temp_ids)
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|sov|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eov|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def i2v_prompt(self, image_ids, video_ids):
        """
        :param image_ids:
        :param video_ids:
        :return:
        """
        pass

    def lvg_prompt(self, text_ids, image_ids, labels):

        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = torch.rand(len(text_ids))
        probs2 = torch.rand(len(text_ids))
        for i in range(len(text_ids)):

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [int(self.sptids_dict['<|t2i|>']), self.text_tokenizer.bos_token_id,
                            self.text_tokenizer.eos_token_id]

            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * (len(temp_ids) + image_ids.shape[-1] + 3)
            else:
                # should add the eos token
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                # should we predict text tokens when doing image reconstruction?
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                labels[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0), torch.cat(label_ids, dim=0)

    def lvg_gen_prompt(self, text_ids, image_ids):
        device = image_ids.device
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict['<|t2i|>'])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - len(temp_ids)) + [1] * len(temp_ids)
            else:
                temp_ids = temp_ids[:self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = torch.cat([
                torch.tensor(temp_ids).to(device),
                self.sptids_dict['<|soi|>'].to(device),
                image_ids[i],
                self.sptids_dict['<|eoi|>'].to(device)
            ], dim=0)

            temp_masks = torch.tensor(temp_masks).to(device)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return torch.cat(sequence_ids, dim=0), torch.cat(attention_masks, dim=0)

    def mask_prompt(self):
        pass

    def mmu_conv(self, images_embeddings, input_ids, label_ids, input_ids_system):
        device = input_ids.device
        discrete_image_input = images_embeddings.ndim == 2
        img_seq_len = images_embeddings.shape[1]
        if label_ids is None:
            label_ids = copy.deepcopy(input_ids)

        input_ids_part1 = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|soi|>']).to(device)], dim=1).long()
    
        input_ids_part2 =  torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.sptids_dict['<|eoi|>']).to( device),
            input_ids], dim=1).long()
        
        if input_ids_system is not None:
            input_ids_part1 = torch.cat([input_ids_system, input_ids_part1], dim=1).long()
            label_ids = torch.cat([
                torch.ones_like(input_ids_system) * self.ignore_id,  # ignore system prompt
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], img_seq_len) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                label_ids.to(device)
            ], dim=1).long()
        else:
            label_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], img_seq_len) * self.ignore_id).to(device),
                (torch.ones(input_ids.shape[0], 1) * self.ignore_id).to(device),
                label_ids.to(device)
            ], dim=1).long()
        
        if discrete_image_input:
            input_ids = torch.cat([input_ids_part1, images_embeddings, input_ids_part2], dim=1).long()
            return input_ids, None, label_ids
        else:
            return input_ids_part1, input_ids_part2, None, label_ids
            
    def mmu_embed(self, images_embeddings, text_ids):
        device= images_embeddings.device
        img_seq_len = images_embeddings.shape[1]
        prefix_ids = []
        suffix_ids = []
        label_ids = []
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
                
            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]
            max_text_len = self.max_text_len - 1
            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.pad_id] * (max_text_len - len(temp_ids))
            else:
                # should add the eos token
                temp_ids = temp_ids[:max_text_len - 1] + [self.text_tokenizer.eos_token_id]

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = torch.cat([
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor([self.ignore_id]).to(device),
                torch.ones_like(images_embeddings[:, :, 0]) * self.ignore_id,
                torch.tensor([self.ignore_id]).to(device),
                torch.tensor(temp_ids).to(device),
            ], dim=0)

            temp_label_ids = torch.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)
            temp_ids = torch.cat([
                self.sptids_dict['<|mmu|>'].to(device),  # task token
                self.sptids_dict['<|soi|>'].to(device),
            ], dim=0)
            suffix_temp_ids = torch.cat([self.sptids_dict['<|eoi|>'].to(device), 
                                         torch.tensor(temp_ids).to(device)], dim=0)
            suffix_ids.append(suffix_temp_ids.unsqueeze(0))
            prefix_ids.append(temp_ids.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return torch.cat(prefix_ids, dim=0), torch.cat(suffix_ids, dim=0), None, torch.cat(label_ids, dim=0)
        
    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])

        elif task == "t2v":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2v_prompt(text_ids, image_ids, input[2])

        elif task == "t2i_plus_lm":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids[:config.training.batch_size], image_ids,
                                                                   input[2])
            sequence_ids_with_masks_lm = self.lm_prompt(text_ids[config.training.batch_size:], input[3])
            return sequence_ids_with_masks, sequence_ids_with_masks_lm

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            max_len = None if len(input) == 2 else input[2]
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids, image_ids, max_len)

        elif task == "t2v_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2v_gen_prompt(text_ids, image_ids)

        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)['input_ids']  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])

        elif task == "mmu":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])['input_ids']
            sequence_ids_with_masks = self.mmu_prompt(image_ids, text_ids)

        elif task == "t2v":
            text_ids = self.text_tokenizer(input[0]['input_ids'])
            video_ids = self.vision_tokenizer(input[1])
            sequence_ids_with_masks = self.t2v_prompt(text_ids, video_ids)

        elif task == "i2v":
            image_ids = self.text_tokenizer(input[0])
            video_ids = self.vision_tokenizer(input[1])
            sequence_ids_with_masks = self.i2v_prompt(image_ids, video_ids)

        elif task == "lvg":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.lvg_prompt(text_ids, image_ids, input[2])

        elif task == "lvg_gen":
            text_ids = self.text_tokenizer(input[0])['input_ids']  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.lvg_gen_prompt(text_ids, image_ids)
        
        elif task == "mmu_conv":
            sequence_ids_with_masks = self.mmu_conv(input[0],  input[1], input[2], input[3])
        
        elif task == "mmu_emb":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])['input_ids']
            sequence_ids_with_masks = self.mmu_embed(image_ids, text_ids)
            
        else:
            raise NotImplementedError

        return sequence_ids_with_masks

def create_attention_mask_predict_next(sequence, pad_id=128256, soi_id=128257, eoi_id=128258, rm_pad_in_image=False,
                                       return_inverse_mask=True):
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Masks to identify different types of tokens
    is_padding = sequence == pad_id

    is_start_image = sequence == soi_id

    is_end_image = sequence == eoi_id

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)
    cumulative_end = torch.cumsum(is_end_image, dim=1)
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image

    is_text = ~(in_image_segment)

    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    mask_text = is_text[:, :, None] * causal_mask[None, :, :]

    is_text_image = is_text | in_image_segment

    mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :]
    if rm_pad_in_image:
        sid_img = torch.where(sequence == soi_id)[1]
        for i in range(mask_text_image_bi.shape[0]):
            pad_end_idx = torch.where(sequence[i] == pad_id)
            if len(pad_end_idx[0]) != 0:
                pad_end_idx = pad_end_idx[0][-1]
                mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
            id_padding = torch.where(is_padding[i] == True)
            mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0

    mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text.unsqueeze(1)


def create_attention_mask_for_mmu(sequence, eoi_id=128258, return_inverse_mask=True):
    N, L = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    eoi_pos = torch.where(sequence == eoi_id)[1][0]
    causal_mask[:, :, :, :eoi_pos + 1] = 1

    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask
    else:
        return causal_mask

def create_attention_mask_for_mmu_vit(
        sequence,
        return_inverse_mask=True,
        return_causal_mask=False,
        system_prompt_len=0,
        num_images=1,
        num_tokens=576,
        prefix_length=-1,
        eos_pos=None,
):
    N, L, H = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    # TODO: a simple solution for multi-image input 
    
    if not return_causal_mask:
        if prefix_length > 0:
            start_index = prefix_length
        else:
            start_index = 1+ system_prompt_len +1
        
        if isinstance(num_tokens, int):
            end_index = start_index + num_tokens * num_images # for 336px input
            causal_mask[:, :, :, start_index: end_index] = 1
        else:
            img_lens = (num_tokens[:, 0] * num_tokens[:, 1]).tolist()
            for i in range(N):
                end_index = start_index + img_lens[i]
                causal_mask[i, :, :, start_index: end_index] = 1
                
    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(torch.int64)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(torch.int64).min
        )
        return inverted_mask
    else:
        return causal_mask

if __name__ == '__main__':
    pass