#!/bin/bash
# Training script: Qwen2.5-VL + TOWER_PLACEHOLDER generative encoder (spatial reasoning)

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Path Configuration  (EDIT THESE)
# ======================
MODEL_PATH="data/models/Qwen2.5-VL-7B-Instruct/"
GENERATIVE_ENCODER_PATH="data/models/dinov3-vitl16-pretrain-lvd1689m"
DINO_MODEL_NAME="vit_large_patch16_dinov3"
OUTPUT_DIR="ckpt/gen-dinov3-7b-spatial-reasoning-s1"
CACHE_DIR="./cache"
mkdir -p $OUTPUT_DIR

# ======================
# Dataset & Hyperparameters
# ======================
DATASETS="spar_234k,llava_hound_64k"
LR=1e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 12800 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 50 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_generative_encoder True \
            --generative_encoder_tower_type "dinov3_online" \
            --generative_encoder_path $GENERATIVE_ENCODER_PATH \
            --generative_vision_tower_model_name $DINO_MODEL_NAME \
            --generative_fusion_method "token_gated_residual" \
            --generative_merger_type "mlp"
#            > ${OUTPUT_DIR}/train.log 2>&1

set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")

export LMMS_EVAL_DATA_ROOT=${LMMS_EVAL_DATA_ROOT:-data/evaluation}
if [ -z "${LMMS_EVAL_VSIBENCH_DATASET_PATH:-}" ] && [ -d "${LMMS_EVAL_DATA_ROOT}/VSI-Bench" ]; then
    export LMMS_EVAL_VSIBENCH_DATASET_PATH="${LMMS_EVAL_DATA_ROOT}/VSI-Bench"
fi
if [ -z "${LMMS_EVAL_CVBENCH_DATASET_PATH:-}" ] && [ -d "${LMMS_EVAL_DATA_ROOT}/CV_Bench" ]; then
    export LMMS_EVAL_CVBENCH_DATASET_PATH="${LMMS_EVAL_DATA_ROOT}/CV_Bench"
fi

accelerate launch --num_processes=8 -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$OUTPUT_DIR,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path

set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=cvbench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")

accelerate launch --num_processes=8 -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$OUTPUT_DIR,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path
