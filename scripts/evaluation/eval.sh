set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=zd11024/vgllm-qa-vggt-4b

# Optional local dataset root for offline evaluation.
# If corresponding local folders exist, they are used automatically.
export LMMS_EVAL_DATA_ROOT=${LMMS_EVAL_DATA_ROOT:-data/evaluation}
if [ -z "${LMMS_EVAL_VSIBENCH_DATASET_PATH:-}" ] && [ -d "${LMMS_EVAL_DATA_ROOT}/VSI-Bench" ]; then
    export LMMS_EVAL_VSIBENCH_DATASET_PATH="${LMMS_EVAL_DATA_ROOT}/VSI-Bench"
fi
if [ -z "${LMMS_EVAL_CVBENCH_DATASET_PATH:-}" ] && [ -d "${LMMS_EVAL_DATA_ROOT}/CV_Bench" ]; then
    export LMMS_EVAL_CVBENCH_DATASET_PATH="${LMMS_EVAL_DATA_ROOT}/CV_Bench"
fi

accelerate launch --num_processes=8 -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path
