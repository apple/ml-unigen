#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Started from https://github.com/showlab/Show-o/blob/main/models/modeling_showo.py
# Copyright 2024 HuggingFace, NUS Show Lab.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import Qwen2ForCausalLM
from utils.checkpoint_registry import real_checkpoint
from .multimodal_encoder.builder import get_vision_tower

class UniGen(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_und_encoder: bool,
            vocab_size: int,
            llm_vocab_size: int,
            llm_model_path: str = '',
            codebook_size: int = 8192,
            num_vq_tokens: int = 256,
            load_from_pretrained: bool = True,
            mm_input_dim: int = 1024,
            gen_input_dim: int = 16,
            und_proj_depth: int = 0,
            gen_proj_depth: int = 0,
            use_gen_dim: bool = False,
            rope_theta: Optional[float] = None,
            scaling_factor: float = 1.0,
            rope_type: str = 'linear',
            vision_tower_name: Optional[str] = None,
            ckpt_base_path: str = "",
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_vq_tokens = num_vq_tokens

        if ckpt_base_path:
            llm_model_path = real_checkpoint(llm_model_path, ckpt_base_path)

        config = AutoConfig.from_pretrained(llm_model_path)
        self.register_to_config(hidden_size=config.hidden_size) # for deepspeed auto config
        
        # We can support more LLM in the future
        model_fn = Qwen2ForCausalLM
        
        if load_from_pretrained:
            if vocab_size != config.vocab_size:
                config.vocab_size = vocab_size
            if rope_theta is not None and rope_theta  != config.rope_theta:
                config.rope_theta = rope_theta
            if scaling_factor != 1:
                config.rope_scaling = {"factor": float(scaling_factor), "type": rope_type }
            self.llm = model_fn(config)
        else:
            self.llm = model_fn.from_pretrained(llm_model_path, attn_implementation='sdpa')
            if config.vocab_size != vocab_size:
                self.llm.resize_token_embeddings(vocab_size)

        self.output_size = self.vocab_size
        self.img_output_size  = codebook_size

        if gen_proj_depth > 0:
            if use_gen_dim:
                self.gen_embed = torch.nn.Embedding(codebook_size + 1,  gen_input_dim)
                modules = [torch.nn.Linear(gen_input_dim, config.hidden_size)]
                for _ in range(1, gen_proj_depth):
                    modules.append(torch.nn.GELU())
                    modules.append(torch.nn.Linear(config.hidden_size, config.hidden_size))
                self.gen_projector =torch.nn.Sequential(*modules)
            else:
                self.gen_embed = torch.nn.Embedding(codebook_size + 1, config.hidden_size)
                modules = [torch.nn.Linear( config.hidden_size, config.hidden_size * 2)]
                for _ in range(1, gen_proj_depth):
                    modules.append(torch.nn.GELU())
                    modules.append(torch.nn.Linear(config.hidden_size * 2, config.hidden_size))
                self.gen_projector =torch.nn.Sequential(*modules)
            self.img_head = torch.nn.Linear(config.hidden_size, codebook_size, bias=False)
            self.register_to_config(mask_token_id=codebook_size)
        else:
            self.register_to_config(mask_token_id=vocab_size - 1)
            
        if w_und_encoder:
            if vision_tower_name is not None:
                self.init_vision_tower(vision_tower_name)
            self.add_mm_projector(max(2, und_proj_depth), mm_input_dim)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
    
    def resize_token_embeddings(self, vocab_size):
        self.vocab_size = vocab_size
        self.llm.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size
        
    def init_vision_tower(self, vision_tower_name):
        self.register_to_config(vision_tower_name=vision_tower_name)
        if self.config.ckpt_base_path:
            vision_tower_name =  real_checkpoint(vision_tower_name, self.config.ckpt_base_path)
        self.vision_tower = get_vision_tower(vision_tower_name, freeze=False)

    def add_vision_tower(self, config):
        vision_tower_name = config.model.vision_tower.name
        self.init_vision_tower(vision_tower_name)
        mm_input_dim= self.vision_tower.config.hidden_size if hasattr(self.vision_tower, 'config') else self.vision_tower.hidden_size
        self.add_mm_projector(config.model.unigen.get('und_proj_depth', 2), mm_input_dim)
    
    def add_mm_projector(self, mlp_depth, mm_input_dim):
        self.register_to_config(w_und_encoder=True)
        self.register_to_config(mm_input_dim=mm_input_dim)
        self.register_to_config(und_proj_depth=mlp_depth)
        hidden_size = self.config.hidden_size
        modules = [torch.nn.Linear(mm_input_dim, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(torch.nn.GELU())
            modules.append(torch.nn.Linear(hidden_size, hidden_size))
        self.mm_projector =torch.nn.Sequential(*modules)
        
    def get_gen_embed(self, img_tokens):
        return self.gen_projector(self.gen_embed(img_tokens))

    def prepare_inputs_for_mmu(self,
        image_feats,                # [B, N_max, D]
        spatial_shapes,             # [B, 2]
        input_ids,                  # [B, L_txt]
        label_ids,
        prompt_template,            # tokenizer
        input_ids_system = None,    # [B, L_sys]
    ):
        device = input_ids.device
        batch_size = input_ids.shape[0]
        pad_token_id = prompt_template.text_tokenizer.pad_token_id
        max_seq_len = prompt_template.max_seq_len
        
        image_embeds = self.mm_projector(image_feats)
        
        if label_ids is None:
            label_ids = input_ids.clone()
       
        if prompt_template.task_token_first:
            input_ids_part1 = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|im_start|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|soi|>']).to(device)
            ], dim=1).long()
        else:
            input_ids_part1 = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|im_start|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|soi|>']).to(device)
            ], dim=1).long()
        
        if input_ids_system is not None:
            input_ids_part1 = torch.cat([input_ids_system, input_ids_part1], dim=1)  # [B, L_sys + 3]
        
        valid_img_lens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        max_img_lens = max(valid_img_lens)
        full_img_len = image_embeds.shape[1]
        embed_part1 = self.llm.model.embed_tokens(input_ids_part1)            # [B, L1, D]
        if self.training:
            input_ids_part2 =  torch.cat([
                (torch.ones(input_ids.shape[0], 1) * prompt_template.sptids_dict['<|eoi|>']).to(device),
                input_ids[:, 1:],
                (torch.ones(input_ids.shape[0], full_img_len) * pad_token_id).to(device),
            ], dim=1).long()
            #  generating text embedding
            embed_part2 = self.llm.model.embed_tokens(input_ids_part2)            # [B, L2, D]
        else:
            embed_part2 = []
            for i in range(batch_size):
                L_img = valid_img_lens[i]
                curr_part2 = torch.cat([
                    (torch.ones(1) * prompt_template.sptids_dict['<|eoi|>']).to(device),
                    input_ids[i, 1:],
                    (torch.ones(max_img_lens - L_img) * pad_token_id).to(device),
                ], dim=0).long()
                embed_part2.append(self.llm.model.embed_tokens(curr_part2.unsqueeze(0)).squeeze(0))

        full_embeddings = []
        label_ids_all = []
        for i in range(batch_size):
            L_img = valid_img_lens[i]
            cur = torch.cat([
                embed_part1[i],
                image_embeds[i, :L_img],
                embed_part2[i]], dim=0)
            cur_labels =  torch.cat([
                (torch.ones(input_ids_part1.shape[1]) * prompt_template.ignore_id).to(device),
                (torch.ones(L_img) * prompt_template.ignore_id).to(device),
                (torch.ones(1) * prompt_template.ignore_id).to(device),
                label_ids[i, 1:].to(device),
                (torch.ones(full_img_len) * prompt_template.ignore_id).to(device),
                ], dim=0).long()
            full_embeddings.append(cur[:max_seq_len])
            label_ids_all.append(cur_labels[:max_seq_len])
        
        full_embeddings = torch.stack(full_embeddings, dim=0)
        label_ids_all = torch.stack(label_ids_all, dim=0)
        label_ids_all[label_ids_all == pad_token_id] = prompt_template.ignore_id
        
        attention_mask = torch.zeros((batch_size, max_seq_len), device=device).to(torch.bool)
        position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        eoi_pos = torch.where(label_ids_all.flip(-1) == prompt_template.eos_token_id) # find the first pad token
        eoi_id, eoi_len = 0, len(eoi_pos[0])
        for i in range(batch_size):
            if eoi_id < eoi_len and eoi_pos[0][eoi_id] == i:
                cur_len = label_ids_all.shape[1] - eoi_pos[1][eoi_id] 
                while eoi_id + 1 < eoi_len and eoi_pos[0][eoi_id + 1 ] == i:
                    eoi_id +=1
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
            else:
                cur_len = label_ids_all.shape[1]
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=device)
                
        return full_embeddings, attention_mask, label_ids_all, input_ids_part1

    def prepare_inputs_for_t2i(self,
        input_ids,        # [B, N_max]
        num_vq_tokens,
    ):
        input_embeddings = self.llm.model.embed_tokens(input_ids)  
        if self.config.get('gen_proj_depth', 0) > 0:
            out_image = self.get_gen_embed(input_ids[:, -(num_vq_tokens + 1): -1].contiguous())
            input_embeddings[:, -(num_vq_tokens + 1): -1] = out_image
        return input_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_smoothing: float = 0.0,
        batch_size_t2i: int = 0,
        batch_size_lm: int = 0,
        batch_size_mmu: int = 0,
        max_seq_length: int = 128,
        num_vq_tokens: int = 256,
        t2i_mode: str = 'mask',
        **kwargs,
    ):
        if self.config.get('gen_proj_depth', 0) > 0 and batch_size_t2i > 0:
            if input_embeddings is None:
                input_embeddings = self.llm.model.embed_tokens(input_ids)
                image_embed = self.get_gen_embed(input_ids[:, -(num_vq_tokens + 1): -1].contiguous())
                input_embeddings[:, -(num_vq_tokens + 1): -1] = image_embed
            outputs = self.llm.model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
            img_logits = self.img_head(hidden_states[:batch_size_t2i])
            if labels is None:
                return img_logits
            logits = self.llm.lm_head(hidden_states[batch_size_t2i:])
        else:
            img_logits = None
            if input_embeddings is None:
                outputs = self.llm.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True
                )
            else:
                outputs = self.llm.model(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )
            hidden_states = outputs.last_hidden_state
            logits = self.llm.lm_head(hidden_states)

        if labels is None:
            return logits
        else:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            # loss_t2i = F.cross_entropy(
            #     logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            #     labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            if img_logits is not None:
                if t2i_mode == 'mask':
                    loss_t2i = F.cross_entropy(
                        img_logits[:, -(num_vq_tokens + 1): -1].contiguous().view(-1, self.img_output_size),
                        labels[:batch_size_t2i, -(num_vq_tokens + 1): -1].contiguous().view(-1), ignore_index=-100,
                    )
                elif t2i_mode == 'ar':
                    loss_t2i = F.cross_entropy(
                        img_logits[:, -(num_vq_tokens + 2): -1].contiguous().view(-1, self.img_output_size),
                        labels[:batch_size_t2i,  -(num_vq_tokens + 1): ].contiguous().view(-1), ignore_index=-100,
                    )
                batch_size_lm_start = 0
            else:
                if t2i_mode == 'mask':
                    loss_t2i = F.cross_entropy(
                        logits[:batch_size_t2i, -(num_vq_tokens + 1): -1].contiguous().view(-1, self.output_size),
                        labels[:batch_size_t2i, -(num_vq_tokens + 1): -1].contiguous().view(-1), 
                        ignore_index=-100,
                    )
                elif t2i_mode == 'ar':
                     loss_t2i = F.cross_entropy(
                        logits[:batch_size_t2i, -(num_vq_tokens + 2): -1].contiguous().view(-1, self.output_size),
                        labels[:batch_size_t2i,-(num_vq_tokens + 1): ].contiguous().view(-1),
                        ignore_index=-100,
                    )
                batch_size_lm_start = batch_size_t2i

            # 2. Next token prediction for language modeling
            loss_lm = 0.
            if batch_size_lm > 0:
                loss_lm = F.cross_entropy(
                    logits[batch_size_lm_start: batch_size_lm_start + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                    labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
                )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = 0.
            if batch_size_mmu > 0:
                loss_mmu = F.cross_entropy(
                    logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                    labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
                )
            if img_logits is not None:
                return img_logits, loss_t2i, loss_lm, loss_mmu
            else:
                return logits, loss_t2i, loss_lm, loss_mmu

    def t2i_generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        uncond_input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.Tensor] = None,
        uncond_input_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        timesteps: int = 18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale: int = 0,
        noise_schedule = cosine_schedule,
        generator: Optional[torch.Generator] = None,
        image_token_num_per_image: int = 256,
        text_vocab_size: int = 151936,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        
        input_ids_minus_lm_vocab_size = input_ids[:, -(image_token_num_per_image + 1):-1].clone()
  
        if input_embeddings is None:
            input_embeddings = self.llm.model.embed_tokens(input_ids)
        
        if self.config.get('gen_proj_depth', 0) > 0:
            image_embeddings = self.get_gen_embed(input_ids[:, -(image_token_num_per_image + 1): -1].contiguous())
        else:
            image_embeddings = input_embeddings[:, -(image_token_num_per_image + 1): -1]
        
        bsz = image_embeddings.shape[0]
        prefix_embedding = input_embeddings[:, :-(image_token_num_per_image + 1)]
        suffix_embedding = input_embeddings[:, -1:]
        
        #prepare for uncond input
        if guidance_scale > 1:
            if uncond_input_embeddings is None: #and uncond_input_ids is not None:
                prefix_embedding = torch.cat([
                    prefix_embedding,
                    self.llm.model.embed_tokens(uncond_input_ids[:, :-(image_token_num_per_image + 1)])
                ])
            else:
                prefix_embedding = torch.cat([
                    prefix_embedding,
                    uncond_input_embeddings[:, :-(image_token_num_per_image + 1)]
                ])
            repeat_n = 2
            suffix_embedding = torch.cat([suffix_embedding] * (repeat_n))
        else:
            repeat_n = 1

        for step in range(timesteps):
            image_embeddings = torch.cat([image_embeddings]* (repeat_n))
            input_embeddings = torch.cat([prefix_embedding, image_embeddings, suffix_embedding], 1)
            if self.config.get('gen_proj_depth', 0) > 0:
                outputs = self.llm.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, return_dict=True)
                hidden_states = outputs.last_hidden_state
                output = self.img_head(hidden_states)
            else:
                output = self(input_ids=input_ids, input_embeddings=input_embeddings, attention_mask=attention_mask) 
            if repeat_n > 1:
                cond_logits = output[:bsz]
                uncond_logits = output[bsz:]
                logits = guidance_scale * (cond_logits-uncond_logits[:bsz]) + uncond_logits[:bsz]
            else:
                logits = output
            if self.config.get('gen_proj_depth', 0) > 0:
                logits = logits[:, -(image_token_num_per_image + 1):-1, ]
            else:
                logits = logits[:, -(image_token_num_per_image + 1):-1, text_vocab_size:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))

            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            
            mask_len = (image_token_num_per_image * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device),
                torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            if self.config.get('gen_proj_depth', 0) > 0:
                input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
                image_embeddings = self.get_gen_embed(input_ids_minus_lm_vocab_size)
            else:
                input_image_ids = torch.where(masking, mask_token_id, sampled_ids + text_vocab_size)
                image_embeddings = self.llm.model.embed_tokens(input_image_ids)
                input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    def t2i_generate_ar(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        uncond_input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.Tensor] = None,
        uncond_input_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_scale: int = 0,
        temperature: float = 1.0,
        text_vocab_size: int = 151936,
        image_token_num_per_image: int = 256,
        **kwargs,
    ):
        generated_tokens = torch.zeros((input_ids.shape[0], image_token_num_per_image), dtype=torch.int).cuda()
        if uncond_input_embeddings is not None:
            uncond_input_embeddings = uncond_input_embeddings[:, : -(image_token_num_per_image + 1)]
        elif uncond_input_ids is not None:
            uncond_input_ids = uncond_input_ids[:, : -(image_token_num_per_image + 1)]
        if input_embeddings is not None:
            bsz = input_embeddings.shape[0]
            input_embeddings = input_embeddings[:,  :-(image_token_num_per_image + 1)]
            model_input = torch.cat([input_embeddings, uncond_input_embeddings])
        else:
            bsz = input_ids.shape[0]
            input_ids = input_ids[:,  :-(image_token_num_per_image + 1)]
            model_input = torch.cat([input_ids, uncond_input_ids])
        curr_seq_len = model_input.shape[1]
       
        for i in range(image_token_num_per_image):
            if self.config.get('gen_proj_depth', 0) > 0:
                outputs = self.llm.model(
                    inputs_embeds=model_input,
                    attention_mask=attention_mask[:, :curr_seq_len],
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None
                )
                hidden_states = outputs[0]
                logits = self.img_head(hidden_states[:, -1, :])
            else:
                outputs = self.llm(
                    inputs_embeds=model_input,
                    attention_mask=attention_mask[:, :curr_seq_len],
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None,
                    num_logits_to_keep=1
                )
                logits = outputs.logits # bs*2, 1, vocab_size
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + guidance_scale * (cond_logits-uncond_logits)
            if self.config.get('gen_proj_depth', 0) == 0:
                logits = logits[:, -1, text_vocab_size:-1]
            
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # bs, 1
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            if self.config.get('gen_proj_depth', 0) > 0:
                next_token = torch.cat([next_token, next_token]) 
                model_input = self.gen_projector(self.gen_embed(next_token))
            else:
                next_token = torch.cat([next_token, next_token]) + text_vocab_size # bs*2, 1
                model_input =  self.llm.model.embed_tokens(next_token)
            curr_seq_len +=1
            
        return generated_tokens
    
    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            # FIXME: a temporary solution for transformers > 4.41
            attention_mask = attention_mask_b.unsqueeze(0).unsqueeze(0)

            # pluck the logits at the final step and scale by desired temperature
            if temperature > 0:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits[:, -1], dim=-1).reshape(-1, 1)

            result.append(idx_next[0][0])
            if self.config.w_und_encoder:
                idx_next_embeddings = self.llm.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)
            
            if eot_token is not None and idx_next.cpu() == eot_token:
                break
        return result

    @torch.no_grad()
    def generate(self, input_ids=None, input_embeddings=None, attention_mask=None, **kwargs):
        if input_embeddings is None:
            return self.llm.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        else:
            return self.llm.generate(attention_mask=attention_mask, inputs_embeds=input_embeddings, **kwargs)

