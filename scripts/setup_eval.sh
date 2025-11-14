#!/bin/bash

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 2>/dev/null)
echo "[[[ Detect GPU: ${GPU_NAME} ]]]"

# download mmcv-1.7.1
wget -P third_party/geneval/ https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.7.1.tar.gz
tar -xvf third_party/geneval/v1.7.1.tar.gz -C third_party/geneval/ &&  rm third_party/geneval/v1.7.1.tar.gz

# clone mmdetection
git clone -b 2.x https://github.com/open-mmlab/mmdetection.git third_party/mmdetection

# clone dpg-bench
git clone https://github.com/TencentQQGYLab/ELLA.git third_party/ELLA
mv third_party/ELLA/dpg_bench  third_party/dpg_bench 
rm -rf third_party/ELLA

# prepare dpg metadata
python third_party/prepare_dpg_metadata.py

# clone geneval
git clone https://github.com/djghosh13/geneval.git  third_party/geneval

# clone lmms-eval
wget -P third_party/ https://github.com/EvolvingLMMs-Lab/lmms-eval/archive/refs/tags/v0.3.0.tar.gz
tar -xvf third_party/v0.3.0.tar.gz -C third_party/  &&  rm third_party/v0.3.0.tar.gz 

# add unigen to lmms model 
mv third_party/lmms-eval-0.3.0 third_party/lmms-eval
rm -r third_party//lmms-eval/lmms_eval/models && cp -r third_party/lmms/models  third_party/lmms-eval/lmms_eval/models 

if [[ ${GPU_NAME} == *"A100"* ]]; then
  echo "[A100] Installing mmcv + mmdetect"
  # Install mmcv (A100: compute capability 8.0, C++17)
  MMCV_WITH_OPS=1 CXXFLAGS="-std=c++17" MMCV_CUDA_ARGS="-std=c++17" pip install -e third_party/geneval/mmcv-1.7.1/

  # install mmdetection
  pip install -v -e third_party/mmdetection/
elif [[ ${GPU_NAME} == *"H100"* ]]; then
  echo "[H100] Installing mmcv + mmdetect"
  # Install mmcv (H100: compute capability 9.0, C++17)
  MMCV_WITH_OPS=1 CXXFLAGS="-std=c++17" MMCV_CUDA_ARGS="-gencode=arch=compute_90,code=sm_90 -std=c++17" pip install -e third_party/geneval/mmcv-1.7.1/

  # Install mmdetection (explicit arch list for sm_90)
  TORCH_CUDA_ARCH_LIST="9.0" pip install -v -e third_party/mmdetection/
fi

pip install pip==24.0
pip install taming-transformers-rom1504 fairseq librosa==0.10.1 
pip install unicodedata2 zhconv simplejson
pip install datasets==2.16.1 modelscope==1.28.1 
pip install -e third_party/lmms-eval/