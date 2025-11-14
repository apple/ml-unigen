#!/bin/bash
set -e -u
#---------------------------------------------------------------------
# Usage
#---------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $0 --experiment_config CONFIG --output_dir OUTPUT_DIR --pretrained_model PRETRAINED_MODEL

  --experiment_config          Configuration file for SFT stage
  --user_id                   Your personal identifier
  --pretrained_model          Pretrained model path to load PT weights
  --train_module              (Optional) Training module, default=train_w_clip_vit.py
EOF
    exit 1
}

#---------------------------------------------------------------------
# Parse arguments
#---------------------------------------------------------------------
EXP_CONFIG=""
PRETRAINED_MODEL=""
CHECKPOINTS=""
TRAIN_MODULE="train_w_clip_vit.py"
USER_ID=""
DS_CKPT_DIR="unigen_data"
OUTPUT_DIR=''
ACCELERATE_CONFIG_8GPU="configs/accelerate_configs/8_gpus_deepspeed_zero2.yaml"
ACCELERATE_CONFIG_SINGLE_DEEPSPEED="configs/accelerate_configs/deepspeed/zero2.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment_config)
            EXP_CONFIG="$2"
            shift 2;;
        --pretrained_model)
            PRETRAINED_MODEL="$2"
            shift 2;;
        --train_module)
            TRAIN_MODULE="$2"
            shift 2;;
        --user_id)
            USER_ID="$2"
            shift 2;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2;;
        -h|--help)
            usage;;
        *)
            echo "Unknown option: $1" >&2
            usage;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "You haven't provide --output_dir, abort!"
    usage
    exit
else
    mkdir -p ${OUTPUT_DIR}
fi

# Check required parameters
if [[ -z "$EXP_CONFIG" ]]; then
    echo "Error: Missing required arguments --experiment_config" >&2
    usage
    exit
fi

if [[ -z "$PRETRAINED_MODEL" ]]; then
    echo "Error: Missing required arguments --pretrained_model" >&2
    usage
    exit
fi

source scripts/bash_utils.sh

#---------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------
if [[ -n "${PRETRAINED_MODEL}" ]]; then
    load_from_pretrained="True"
    print_main "Initializing SFT from PT checkpoint: ${PRETRAINED_MODEL}"
fi


BASE_MODULE="accelerate launch --config_file ${ACCELERATE_CONFIG_8GPU} --main_process_port=8888 training/${TRAIN_MODULE}"

#---------------------------------------------------------------------
# Stage 3b training
#---------------------------------------------------------------------
RUN_NAME=$(config_path_to_run_name "${EXP_CONFIG}")
STAGE3_NAME="stage-sft-${RUN_NAME}"
cat <<EOF
====================== TASK INFORMATION =======================
  Stage SFT training
  Run name: ${RUN_NAME}
  Stage name: ${STAGE3_NAME}
=========================================================================
EOF

command_stage3p=$(cat << EOF
${BASE_MODULE} config="${EXP_CONFIG}" \
    experiment.name="${STAGE3_NAME}" \
    experiment.output_dir="${OUTPUT_DIR}/${STAGE3_NAME}" \
    model.unigen.pretrained_model_path="${PRETRAINED_MODEL}" \
    data.local_fs="${DS_CKPT_DIR}"
EOF
)
run_command "${command_stage3p}"

