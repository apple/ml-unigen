#!/bin/bash
set -e -u
#---------------------------------------------------------------------
# Usage
#---------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $0 --experiment_config CONFIG --output_dir OUTPUT_DIR [--pretrained-path PRETRAINED_PATH]

  --experiment_config          Configuration file
  --pretrained_model          (Optional) Pretrained model path to load PT weights
  --train_module              (Optional) Training module, default=train.py
EOF
    exit 1
}

#---------------------------------------------------------------------
# Parse arguments
#---------------------------------------------------------------------
EXP_CONFIG=""
PRETRAINED_MODEL=""
CHECKPOINTS=""
TRAIN_MODULE="train.py"
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
    echo "Error: Missing required arguments." >&2
    usage
fi

source scripts/bash_utils.sh

#---------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------
if [[ -z "${PRETRAINED_MODEL}" ]]; then
    print_main "Initializing pre-training from LLM"
    PT_STAGE="1"
    PT_CHECKPOINT_CONFIG=""
    load_from_pretrained="False"
else
    print_main "Initializing pre-training from pre-trained checkpoint: ${PRETRAINED_MODEL}"
    PT_STAGE="2"
    PT_CHECKPOINT_CONFIG="model.unigen.pretrained_model_path=${PRETRAINED_MODEL}"
    load_from_pretrained="True"
fi


BASE_MODULE="accelerate launch --config_file ${ACCELERATE_CONFIG_8GPU} --main_process_port=8888 training/${TRAIN_MODULE}"

#---------------------------------------------------------------------
# Stage 1/2 pre-training
#---------------------------------------------------------------------
RUN_NAME=$(config_path_to_run_name "${EXP_CONFIG}")
STAGE_NAME="stage-${PT_STAGE}-${RUN_NAME}"
cat <<EOF
============================= TASK INFORMATION ============================
  Stage-${PT_STAGE} training
  Run name: ${RUN_NAME}
  Stage name: ${STAGE_NAME}
=========================================================================
EOF

command_stage_pt=$(cat << EOF
${BASE_MODULE} config="${EXP_CONFIG}" \
    experiment.name="${STAGE_NAME}" \
    experiment.output_dir="${OUTPUT_DIR}/${STAGE_NAME}" \
    ${PT_CHECKPOINT_CONFIG:+$PT_CHECKPOINT_CONFIG} \
    data.local_fs="${DS_CKPT_DIR}"
EOF
)
run_command_with_errors "${command_stage_pt}"

