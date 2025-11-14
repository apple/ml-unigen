#!/bin/bash

# install uuidgen
apt update && apt install -y uuid-runtime
python -m pip install --upgrade pip

# setup unigen deps
MAX_JOBS=4 pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/0.43.1/bitsandbytes-0.43.1-py3-none-manylinux_2_24_x86_64.whl
pip install -r scripts/requirements.txt

echo "=========== Setup completed ============"
# accesss API of wandb & huggingface token
wandb login "$WANDB_API_KEY"
huggingface-cli login --token "$HF_TOKEN"

