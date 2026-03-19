#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)


BASE_MODEL=xxxx
CKPT="./ckpt/llavanext-qwen-video3dllm-uniform-lora/checkpoint-10"
ANWSER_FILE="results/multi3drefer/$1.jsonl"
GEN_EXTRA_ARGS=${GEN_EXTRA_ARGS:-}
EXTRA_ARGS=()
if [ -n "$GEN_EXTRA_ARGS" ]; then
    read -r -a EXTRA_ARGS <<< "$GEN_EXTRA_ARGS"
fi

# example: sh scripts/3d/eval/eval_scan2cap_lora.sh $ckpt_name uniform 32
CUDA_VISIBLE_DEVICES=0 python3 llava/eval/model_scan2cap.py \
    --model-path $BASE_MODEL \
    --lora-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 8 \
    --question-file data/processed/scan2cap_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    "${EXTRA_ARGS[@]}"

python llava/eval/eval_scan2cap.py --input-file $ANWSER_FILE
