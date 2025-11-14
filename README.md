# UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation

This project accompanies the research paper,

[UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2505.14682) <br>
[Rui Tian*](https://scholar.google.com/citations?user=zTI-OFoAAAAJ), [Mingfei Gao*](https://fly6464.github.io), [Mingze Xu*](https://xumingze0308.github.io), [Jiaming Hu](https://scholar.google.com/citations?user=vm3imKsAAAAJ&hl=en), [Jiasen Lu](https://jiasenlu.github.io), [Zuxuan Wu](https://zxwu.azurewebsites.net), [Yinfei Yang](https://sites.google.com/site/yinfeiyang), [Afshin Dehghan](https://scholar.google.com/citations?user=wcX-UW4AAAAJ&hl=en)

<p align="center">
    <img src="asset/teaser.png" width="600" alt="The workflow of UniGen using test-time scaling and CoT-V">
</p>

**UniGen** is a unified multimodal large language model (MLLM) capable of both image understanding and generation. We detail UniGen's full training pipeline from a data-centric perspective, including its multi-stage pre-training, supervised fine-tuning, and direct preference optimization.

More importantly, we introduce **Chain-of-Thought Verification (CoT-V)**, a novel test-time strategy that significantly boosts image generation quality using a simple Best-of-N approach.



## 📢 News
- [11/18] 🚀🚀🚀 [UniGen-1.5](https://arxiv.org/abs/2511.14760) is on ArXiv!
- [9/19] 🔥🔥🔥 [UniGen](https://arxiv.org/abs/2505.14682) has been accepted to NeurIPS 2025!



## 📚 Table of Contents
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
    - [Evaluation Installation](#evaluation-installation)
- [License](#license)
- [Citations](#citations)


## 🚀 Getting Started

### Installation

This code requires Python >= 3.10.12, PyTorch >= 2.4.1, and CUDA 12.4.

1.  **[Optional but recommended]** Create and activate a new conda environment.
     ```bash
     conda create -n unigen python=3.10.12
     ```
     And activate the environment.
     ```
     conda activate unigen
     ```
2. Install the required dependencies.
     ```bash
     bash scripts/setup.sh
     ```
3.  Download the pre-trained weights for [Qwen2.5-1.5b](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [MAGViTv2](https://huggingface.co/showlab/magvitv2), and [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) from Hugging Face and place them in the `unigen_data/checkpoints` directory.
    ```bash
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --repo-type model --local-dir unigen_data/checkpoints/Qwen2.5-1.5B-Instruct

    huggingface-cli download showlab/magvitv2 --repo-type model --local-dir unigen_data/checkpoints/magvitv2

    huggingface-cli download google/siglip-so400m-patch14-384 --repo-type model --local-dir unigen_data/checkpoints/siglip-so400m-patch14-384
    ```

4.  Add your OpenAI API key and organization to your environment variables for model evaluation.
    ```bash
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    export OPENAI_ORG="YOUR_OPENAI_ORG"  # Optional
    ```
   

5.  **[Optional]** Add your Weights & Biases API key to enable logging during training.
    ```bash
    wandb login "YOUR_WANDB_API_KEY"
    ```


### Data Preparation

Prepare the following datasets and place them in the `unigen_data/datasets` directory.
1.  **Text-only Dataset**: Download [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) from Hugging Face.
     ```bash
     huggingface-cli download  tiiuae/falcon-refinedweb --repo-type dataset  --local-dir unigen_data/datasets/falcon-refinedweb
     ```
2.  **Image-Text Pair Dataset (for Pre-training)**: Download [CC-12M](https://github.com/google-research-datasets/conceptual-12m), [CC-3M](https://github.com/google-research-datasets/conceptual-captions), [Segment-Anything-11M](https://ai.meta.com/datasets/segment-anything/), and [ImageNet-21K](https://github.com/Alibaba-MIIL/ImageNet21K). Prepare all datasets in the [WebDataset](https://github.com/webdataset/webdataset) format and perform re-captioning using the following system prompt:
     ```
     <|im_start|>system
     You are a helpful assistant.<|im_end|>
     <|im_start|>user
     <|vision_start|><|image_pad|><|vision_end|>
     What is the content of this image?<|im_end|>
     <|im_start|>assistant
     ```
     The re-annotated caption should be saved in the key pf `.txt` in webdataset, and the image should be saved in the key of `.png|.jpg|.jpeg|.wbep`.

3.  **Supervised Fine-Tuning (SFT) Data**:
    * **Generation Data**: Download [JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB) and [text2image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) and prepare them in the [WebDataset](https://github.com/webdataset/webdataset) format.
    * **Understanding Data**: Our paper uses the single-image mixture from [SlowFast-LLaVA-1.5](https://arxiv.org/abs/2503.18943). For this open-source release, we use the [LLaVA-1.5](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) instruction tuning data, as specified in the [training config](./configs/unigen_1_5b/unigen_sft.yaml).

4.  **Direct Preference Optimization (DPO) Data**:
    * Prepare text prompts from various sources.
    * Set up the data annotation environment with `vllm`.
        ```bash
        pip install vllm==0.7.3
        ```
       
    * Convert prompts into related visual questions using an LLM.
        ```bash
        python scripts/dataflows/zeroshot_questions.py --metadata_path /path/to/prompt --out_path /path/to/out --model_name Qwen/Qwen2.5-7B-Instruct
        ```
       
    * Generate N image samples for each text prompt with the UniGen-SFT model.
    * Run the pseudo-labeling pipeline on each image-question pair.
        ```bash
        python scripts/dataflows/zeroshot_vqa.py --metadata_path /path/to/visual_question --out_path /path/to/out --image_root /path/to/img --model_name Qwen/Qwen2.5-VL-7B-Instruct
        ```


## ⚙️ Training Scripts

### Pre-training: Stage 1
Run the following script for Stage 1 pre-training on 2x 80GB H100/A100 GPUs.
```bash
bash scripts/run_pretraining.sh \
     --experiment_config configs/unigen_1_5b/unigen_pt1.yaml \
     --output_dir path_to_your_out \
     --train_module train.py 
```

### Pre-training Stage 2
Place the final checkpoint from Stage 1 (`unigen_pt1/checkpoint-150000`) in unigen_data/checkpoints. Then, run the following command for Stage 2 pre-training on 4x 80GB H100/A100 GPUs.
```bash
bash scripts/run_pretraining.sh \
     --experiment_config configs/unigen_1_5b/unigen_pt2.yaml \
     --pretrained_model  unigen_pt1/checkpoint-150000  \
     --output_dir path_to_your_out \
     --train_module train.py 
```

### Supervised Finetuning
Place the final checkpoint from Stage 2 (`unigen_pt2/checkpoint-400000`) in `unigen_data/checkpoints`. Then, run the following command for SFT on 1x 80GB H100/A100 GPU.
```bash
bash scripts/run_sft.sh \
     --experiment_config configs/unigen_1_5b/unigen_sft.yaml \
     --pretrained_model  unigen_pt2/checkpoint-400000 \
     --train_module train_w_clip_vit.py \
     --output_dir path_to_your_out 
```


### Direct Preference Optimization
Place the final SFT checkpoint (`unigen_sft/checkpoint-145824`) in `unigen_data/checkpoints`. Then, run the following command for DPO on 1x 80GB H100/A100 GPU.
```bash
bash scripts/run_sft.sh \
    --experiment_config configs/unigen_1_5b/unigen_dpo.yaml  \
    --pretrained_model unigen_sft/checkpoint-145824 \
    --train_module train_dpo.py \
    --output_dir path_to_your_out 
```


### CoT-V Post-Training
Place the final DPO checkpoint (`unigen_dpo/unwrapped_model`) in `unigen_data/checkpoints`. Then, run the following command for CoT-V post-training on 1x 80GB H100/A100 GPU.
```bash
bash scripts/run_cotv.sh \
    --experiment_config configs/unigen_1_5b/unigen_cotv_post_sft.yaml \
    --pretrained_model unigen_dpo/unwrapped_model \
    --train_module train_w_clip_vit.py \
    --output_dir path_to_your_out 
```


## Evaluation Scripts

### Evaluation Installation
Install the necessary requirements and clone required repos for evaluating on understanding (lmms-eval) and generation (DPGbench, GenEval) benchmarks.
```bash
bash scripts/setup_eval.sh
```

Next, download the checkpoints required for evaluation.
```bash
LOCAL_CHECKPOINT_DIR=unigen_data/checkpoints
python -c $'from modelscope.hub.snapshot_download import snapshot_download\nsnapshot_download("damo/mplug_visual-question-answering_coco_large_en")'
bash third_party/geneval/evaluation/download_models.sh $LOCAL_CHECKPOINT_DIR
```

### Evaluating UniGen-PT1 Checkpoints
```bash
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_pt1.yaml  \
    --eval_modules geneval+dpgbench \
    --eval_checkpoint unigen_pt1/checkpoint-150000 \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data
```

### Evaluating UniGen-PT2 Checkpoints
```bash
bash scripts/run_evaluation.sh \
     --config  configs/unigen_1_5b/unigen_pt2.yaml \
     --eval_modules geneval+dpgbench \
     --eval_checkpoint unigen_pt2/checkpoint-400000 \
     --output_dir path_to_your_out \
     --local_shared_fs unigen_data
```

### Evaluating UniGen-SFT Checkpoints
```bash
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_sft.yaml  \
    --lmms_tasks "mmmu_val,gqa,ai2d,mme,mathvista_testmini,mmvet" \
    --eval_modules lmms \
    --eval_checkpoint unigen_sft/checkpoint-145824 \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data

bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_sft.yaml  \
    --lmms_tasks "realworldqa,scienceqa_img,seedbench,pope" \
    --eval_modules lmms+geneval+dpgbench \
    --eval_checkpoint unigen_sft/checkpoint-145824 \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data
```

### Evaluating UniGen-DPO Checkpoints
```bash
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_dpo.yaml  \
    --lmms_tasks "mmmu_val,gqa,ai2d,mme,mathvista_testmini,mmvet" \
    --eval_modules lmms \
    --eval_checkpoint unigen_dpo/unwrapped_model \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data
    
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_dpo.yaml  \
    --lmms_tasks "realworldqa,scienceqa_img,seedbench,pope" \
    --eval_modules lmms+geneval+dpgbench \
    --eval_checkpoint unigen_dpo/unwrapped_model \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data
```

### Evaluating UniGen after CoT-V Post-Training
```bash
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_cotv_post_sft.yaml  \
    --lmms_tasks "mmmu_val,gqa,ai2d,mme,mathvista_testmini,mmvet" \
    --eval_modules lmms \
    --eval_checkpoint unigen/checkpoint-500 \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data

bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_cotv_post_sft.yaml  \
    --lmms_tasks "realworldqa,scienceqa_img,seedbench,pope" \
    --eval_checkpoint unigen/checkpoint-500 \
    --output_dir path_to_your_out \
    --local_shared_fs unigen_data
```

### Test-time Scaling of UniGen with CoT-V 
To perform Best-of-N (where N=5) test-time scaling with CoT-V, set mmu_rating_style="think".

A. On the GenEval Benchmark
```bash
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_cotv_post_sft.yaml \
    --eval_modules cot-gen \
    --eval_checkpoint unigen/checkpoint-500 \
    --local_shared_fs unigen_data \
    --output_dir path_to_your_out \
    --mmu_rating_style think
```

B. On the DPG Benchmark
```bash
bash scripts/run_evaluation.sh \
    --config configs/unigen_1_5b/unigen_cotv_post_sft.yaml \
    --eval_modules cot-dpg \
    --eval_checkpoint unigen/checkpoint-500 \
    --local_shared_fs unigen_data \
    --output_dir path_to_your_out \
    --mmu_rating_style think
```


## License
This project is licensed under the [`Apple Sample Code License`](LICENSE).


## Citations
If you are using the data/code/model provided here in a publication, please cite our paper:

	@article{tian2025unigen,
          title={UniGen: Enhanced Training \& Test-Time Strategies for Unified Multimodal Understanding and Generation},
          author={Tian, Rui and Gao, Mingfei and Xu, Mingze and Hu, Jiaming and Lu, Jiasen and Wu, Zuxuan and Yang, Yinfei and Dehghan, Afshin},
          journal={arXiv preprint arXiv:2505.14682},
          year={2025}
          }

	@article{tian2025unigen1.5,
          title={UniGen-1.5: Enhancing Image Generation and Editing through Reward Unification in Reinforcement Learning},
          author={Tian, Rui and Gao, Mingfei and Gang, Haiming and Lu, Jiasen and Gan, Zhe and Yang, Yinfei and Wu, Zuxuan and Dehghan, Afshin},
          journal={arXiv preprint arXiv:2511.14760},
          year={2025}
          }