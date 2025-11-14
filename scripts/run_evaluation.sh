#!/bin/bash
set -e -u
#---------------------------------------------------------------------
# Usage
#---------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $0 --config CONFIG --output_dir OUTPUT_DIR --eval_checkpoint EVAL_CHECKPOINT --eval_modules EVAL_MODULES --lmms_tasks TASKS --mmu_rating_style MMU_RATING_STYLE

  --config                     The config YAML file
  --output_dir                The output directory
  --eval_checkpoint           The checkpoint for evaluation
  --local_shared_fs           The shared base directory that contains checkpoints
  --eval_modules              List of evaluation module names, split by comma, default: lmms+geneval, choices: lmms, geneval, cot-gen, cot-dpg
  --lmms_tasks                List of evaluation task names for LMMS eval, split by comma
  --mmu_rating_style          Set the test time scaling rating prompt style:  single or breakdown, default: single
EOF
    exit 1
}

#---------------------------------------------------------------------
# Parse arguments
#---------------------------------------------------------------------
EVAL_MODULES="lmms+geneval"
EVAL_CHECKPOINT=""
LOCAL_SHARED_FS="unigen_data"
LMMS_TASKS="pope,mmmu_val,seedbench,gqa,scienceqa_img,ai2d,realworldqa,mathvista_testmini"
MMU_RATING_STYLE="single"
OUTPUT_DIR=''
CONFIG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
        CONFIG="$2"
        shift 2;;
    --eval_modules)
        EVAL_MODULES="$2"
        shift 2;;
    --eval_checkpoint)
        EVAL_CHECKPOINT="$2"
        shift 2;;
    --local_shared_fs)
        LOCAL_SHARED_FS="$2"
        shift 2;;
    --lmms_tasks)
        LMMS_TASKS="$2"
        shift 2;;
    --mmu_rating_style)
        MMU_RATING_STYLE="$2"
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

# Check required parameters
if [[ -z "$EVAL_MODULES" || -z "$EVAL_CHECKPOINT" || -z "$OUTPUT_DIR" ]]; then
  echo "Error: Missing required arguments." >&2
  usage
  exit
fi
#---------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------
source scripts/bash_utils.sh

EXP_NAME=$(config_path_to_run_name "${CONFIG}")
OUTPUT_DIR="${OUTPUT_DIR}/${EXP_NAME}_eval_results"
mkdir -p ${OUTPUT_DIR}

RUNNER="accelerate launch --num_processes=8"
export HF_HOME="${LOCAL_SHARED_FS}/datasets/hf_home"

cat <<EOF
============================= TASK INFORMATION ============================
  Evaluate with ${EVAL_MODULES}
  Evaluate on checkpoint: ${EVAL_CHECKPOINT}
  Using runner: ${RUNNER}
  HF_HOME: ${HF_HOME}
  Output dir: ${OUTPUT_DIR}
=========================================================================
EOF


#---------------------------------------------------------------------
# LMMS eval
#---------------------------------------------------------------------
if [[ ${EVAL_MODULES} == *"lmms"* ]]; then
    print_main "--> Evaluating with LMMS EVAL"
    mkdir -p "${OUTPUT_DIR}/lmms_eval"
eval_command=$(cat<<EOF
${RUNNER} -m lmms_eval \
  --model unigen \
  --model_args config="${CONFIG},pretrained=${LOCAL_SHARED_FS}/checkpoints/${EVAL_CHECKPOINT},ckpt_base_path=${LOCAL_SHARED_FS}/checkpoints" \
  --tasks "${LMMS_TASKS}" \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_DIR}/lmms_eval"
EOF
)
  run_command "${eval_command}"
fi

#---------------------------------------------------------------------
# Standard geneval
#---------------------------------------------------------------------
if [[ ${EVAL_MODULES} == *"geneval"* ]]; then
  print_main "--> Evaluating with geneval"
  GENEVAL_ROOT_DIR="third_party/geneval"
  GENEVAL_RESULT_FOLDER="${OUTPUT_DIR}/geneval"
  mkdir -p "${GENEVAL_RESULT_FOLDER}"


eval_command=$(cat<<EOF
${RUNNER} evaluation/inference_geneval.py \
  config=${CONFIG} \
  model.unigen.pretrained_model_path=${EVAL_CHECKPOINT} \
  model.local_checkpoints=${LOCAL_SHARED_FS}/checkpoints \
  experiment.output_dir=${GENEVAL_RESULT_FOLDER} \
  dataset.validation_prompts_file=${GENEVAL_ROOT_DIR}/prompts/evaluation_metadata.jsonl \
  training.guidance_scale=6 training.generation_timesteps=50 \
  inference.eval_text_len=128 \
  inference.n_samples=4
EOF
)
  run_command "${eval_command}"

  # mask2former should be in shared_fs/checkpoints/mask2former
  python ${GENEVAL_ROOT_DIR}/evaluation/evaluate_images.py \
     "${GENEVAL_RESULT_FOLDER}/t2i_samples_step50_scale6" \
     --outfile "${GENEVAL_RESULT_FOLDER}/t2i_samples_step50_scale6/results.jsonl" \
     --model-path "${LOCAL_SHARED_FS}/checkpoints"

  python ${GENEVAL_ROOT_DIR}/evaluation/summary_scores.py \
    "${GENEVAL_RESULT_FOLDER}/t2i_samples_step50_scale6/results.jsonl" >> "${GENEVAL_RESULT_FOLDER}/geneval_score.log"
fi


#---------------------------------------------------------------------
# Standard dpgbench
#---------------------------------------------------------------------
if [[ ${EVAL_MODULES} == *"dpgbench"* ]]; then
  print_main "--> Evaluating with dpgbench"
  DPG_ROOT_DIR="third_party/dpg_bench"
  DPG_RESULT_FOLDER="${OUTPUT_DIR}/dpg_bench"
  mkdir -p "${DPG_RESULT_FOLDER}"

eval_command=$(cat<<EOF
${RUNNER} evaluation/inference_dpg.py \
  config=${CONFIG} \
  model.unigen.pretrained_model_path=${EVAL_CHECKPOINT} \
  model.local_checkpoints=${LOCAL_SHARED_FS}/checkpoints \
  experiment.output_dir=${DPG_RESULT_FOLDER} \
  dataset.validation_prompts_file=${DPG_ROOT_DIR}/dpg_metadata.jsonl \
  training.guidance_scale=6 training.generation_timesteps=50 \
  inference.eval_text_len=256 \
  inference.n_samples=4 
EOF
)
  run_command "${eval_command}"

  # mask2former should be in shared_fs/checkpoints/mask2former
  accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
    ${DPG_ROOT_DIR}/compute_dpg_bench.py \
    --image-root-path ${DPG_RESULT_FOLDER}/dpg_bench_step50_scale6 \
    --csv ${DPG_ROOT_DIR}/dpg_bench.csv  \
    --resolution 256 \
    --pic-num 4 \
    --vqa-model mplug
fi

#---------------------------------------------------------------------
# Test time scaling geneval + self rating
#---------------------------------------------------------------------
if [[ ${EVAL_MODULES} == "cot-gen" ]]; then
  print_main "--> Evaluating with geneval with test time scaling..."
  GENEVAL_ROOT_DIR="third_party/geneval"
  GENEVAL_RESULT_FOLDER="${OUTPUT_DIR}/geneval_cot" #${MMU_RATING_STYLE}"
  mkdir -p "${GENEVAL_RESULT_FOLDER}"

eval_command=$(cat<<EOF
${RUNNER} evaluation/inference_unigen_cot.py \
  config=${CONFIG} \
  model.unigen.pretrained_model_path=${EVAL_CHECKPOINT} \
  model.local_checkpoints=${LOCAL_SHARED_FS}/checkpoints \
  experiment.output_dir=${GENEVAL_RESULT_FOLDER} \
  dataset.validation_prompts_file=${GENEVAL_ROOT_DIR}/prompts/evaluation_metadata.jsonl \
  training.guidance_scale=6 training.generation_timesteps=50 \
  generated_images_dir=${GENEVAL_RESULT_FOLDER} \
  inference.n_samples=20 \
  inference.mmu_prompt_style=${MMU_RATING_STYLE} \
  inference.eval_text_len=128 \
  model.max_new_tokens=128
EOF
)
  run_command "${eval_command}"

  for instance_dir in ${GENEVAL_RESULT_FOLDER}/t2i_samples_step50_scale6_selected_prompt_*/;
  do
    T2I_SELECTED_DIR="$instance_dir"
    echo "evaluating of ${T2I_SELECTED_DIR}"
    folder_name=$(basename "$instance_dir")
    prompt_type=${folder_name#t2i_samples_step50_scale6_selected_prompt}
    echo
    # mask2former should be in shared_fs/checkpoints/mask2former
    python ${GENEVAL_ROOT_DIR}/evaluation/evaluate_images.py \
      "${T2I_SELECTED_DIR}" \
      --outfile "${T2I_SELECTED_DIR}/results.jsonl" \
      --model-path "${LOCAL_SHARED_FS}/checkpoints/mask2former"

    python ${GENEVAL_ROOT_DIR}/evaluation/summary_scores.py \
      "${T2I_SELECTED_DIR}/results.jsonl" >> "${GENEVAL_RESULT_FOLDER}/geneval_score_selected${prompt_type}.log"
  done
  # pick the first 4 as reference as without test time scaling
  T2I_ORIGIN_DIR="${GENEVAL_RESULT_FOLDER}/t2i_samples_step50_scale6_origin"
  mkdir -p ${T2I_ORIGIN_DIR}
  for instance_dir in ${GENEVAL_RESULT_FOLDER}/t2i_samples_step50_scale6/*/;
  do
    instance_id=$(basename "$instance_dir")
    mkdir -p "${T2I_ORIGIN_DIR}/${instance_id}/samples"
    cp "$instance_dir/metadata.jsonl" "${T2I_ORIGIN_DIR}/${instance_id}/"
    cp $instance_dir/samples/0000[0-3].png ${T2I_ORIGIN_DIR}/${instance_id}/samples/
  done

  python ${GENEVAL_ROOT_DIR}/evaluation/evaluate_images.py \
     "${T2I_ORIGIN_DIR}" \
     --outfile "${T2I_ORIGIN_DIR}/results.jsonl" \
     --model-path "${LOCAL_SHARED_FS}/checkpoints/mask2former"

  python ${GENEVAL_ROOT_DIR}/evaluation/summary_scores.py \
    "${T2I_ORIGIN_DIR}/results.jsonl" >> "${GENEVAL_RESULT_FOLDER}/geneval_score_origin.log"

fi


#---------------------------------------------------------------------
# Test time scaling dpg + self rating
#---------------------------------------------------------------------
if [[ ${EVAL_MODULES} == "cot-dpg" ]]; then
  print_main "--> Evaluating with dpgbench with test time scaling..."
  DPG_ROOT_DIR="third_party/dpg_bench"
  DPG_RESULT_FOLDER="${OUTPUT_DIR}/dpg_bench_cot" #${MMU_RATING_STYLE}"
  mkdir -p "${DPG_RESULT_FOLDER}"
  export NCCL_TIMEOUT=4800
  
eval_command=$(cat<<EOF
${RUNNER} evaluation/inference_unigen_cot.py \
  config=${CONFIG} \
  model.unigen.pretrained_model_path=${EVAL_CHECKPOINT} \
  model.local_checkpoints=${LOCAL_SHARED_FS}/checkpoints \
  experiment.output_dir=${DPG_RESULT_FOLDER} \
  dataset.validation_prompts_file=${DPG_ROOT_DIR}/dpg_metadata.jsonl \
  dataset.question_file=${DPG_ROOT_DIR}/dpg_bench_questions.jsonl \
  training.guidance_scale=6 training.generation_timesteps=50 \
  generated_images_dir=${DPG_RESULT_FOLDER} \
  inference.n_samples=20 \
  inference.benchmark=dpg \
  inference.mmu_prompt_style=${MMU_RATING_STYLE} \
  inference.eval_text_len=256 \
  model.max_new_tokens=256
EOF
)
  run_command "${eval_command}"

  for instance_dir in ${DPG_RESULT_FOLDER}/t2i_samples_step50_scale6_selected_*/; 
  do
    T2I_SELECTED_DIR="$instance_dir"
    echo "evaluating of ${T2I_SELECTED_DIR}"
    folder_name=$(basename "$instance_dir")
    prompt_type=${folder_name#t2i_samples_step50_scale6_selected_}
    # mask2former should be in shared_fs/checkpoints/mask2former
    accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
      ${DPG_ROOT_DIR}/compute_dpg_bench.py \
      --image-root-path $instance_dir \
      --csv ${DPG_ROOT_DIR}/dpg_bench.csv  \
      --res-path ${DPG_RESULT_FOLDER}/dpg_bench_${prompt_type}.txt \
      --resolution 256 \
      --pic-num 4 \
      --vqa-model mplug
  done
fi
