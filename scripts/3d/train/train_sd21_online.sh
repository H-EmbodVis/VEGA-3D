#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

# Set up the data folder
IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
DATA_YAML="scripts/3d/train/multi.yaml" # e.g exp.yaml
FRAME_SAMPLING_STRATEGY="uniform"
SD21_BASE_CKPT_DIR="data/models/stable-diffusion-2-1-base"
SD21_EMPTY_PROMPT_PATH="${SD21_BASE_CKPT_DIR}/empty_prompt_embeds.pt"

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
############### Show Envs ####################

nvidia-smi

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="${RUN_NAME:-llavanext-qwen-video3dllm-uniform-sd21-online}"
GENERATIVE_PROFILE="${GENERATIVE_PROFILE:-True}"
PREV_STAGE_CHECKPOINT="data/models/LLaVA-Video-7B-Qwen2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"

NUM_GPUS=8
BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE / NUM_GPUS))
RUN_EVAL=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [ ! -f "${SD21_BASE_CKPT_DIR}/unet/config.json" ]; then
    echo "Missing ${SD21_BASE_CKPT_DIR}/unet/config.json"
    exit 1
fi

if [ ! -f "${SD21_EMPTY_PROMPT_PATH}" ]; then
    echo "Missing ${SD21_EMPTY_PROMPT_PATH}"
    echo "Run: python3 scripts/3d/preprocessing/extract_sd21_empty_prompt_embeds.py --checkpoint_dir ${SD21_BASE_CKPT_DIR} --output_path ${SD21_EMPTY_PROMPT_PATH}"
    exit 1
fi

torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port 43005 \
    llava/train/train_3d.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --embodiedscan_folder data/embodiedscan/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./ckpt/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy $FRAME_SAMPLING_STRATEGY \
    --frames_upbound 32 \
    --use_generative_feature True \
    --generative_feat_dim 1280 \
    --generative_projector_type mlp2x_gelu \
    --generative_feature_source online \
    --generative_vision_tower_type sd21_online \
    --generative_vision_tower_checkpoint $SD21_BASE_CKPT_DIR \
    --generative_vision_tower_empty_prompt_path $SD21_EMPTY_PROMPT_PATH \
    --generative_vision_tower_input_size 896 \
    --generative_vision_tower_timestep 250 \
    --generative_vision_tower_num_inference_steps 1 \
    --generative_vision_tower_chunk_size 16 \
    --generative_vision_tower_output_spatial 14 \
    --generative_cache_max_items -1 \
    --generative_profile $GENERATIVE_PROFILE \
    --use_feat_fusion True \
    --feature_fusion_method token_gated \
    2>&1 | tee -a "./ckpt/${RUN_NAME}.log"

if [ "${RUN_EVAL}" = "1" ]; then
    bash scripts/3d/eval/eval_scanrefer.sh "$RUN_NAME" "$FRAME_SAMPLING_STRATEGY" 32 None
    bash scripts/3d/eval/eval_multi3drefer.sh "$RUN_NAME" "$FRAME_SAMPLING_STRATEGY" 32 None
    bash scripts/3d/eval/eval_sqa3d.sh "$RUN_NAME" "$FRAME_SAMPLING_STRATEGY" 32 None
    bash scripts/3d/eval/eval_scan2cap.sh "$RUN_NAME" "$FRAME_SAMPLING_STRATEGY" 32 None
    bash scripts/3d/eval/eval_scanqa.sh "$RUN_NAME" "$FRAME_SAMPLING_STRATEGY" 32 None
fi
