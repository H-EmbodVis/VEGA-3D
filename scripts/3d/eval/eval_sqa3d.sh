#!/bin/bash

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$(pwd)


CKPT="./ckpt/$1"
ANWSER_FILE="results/sqa3d/$1.jsonl"
if [ -z "$4" ]; then
    echo "Usage: $0 <ckpt_name> <frame_sampling_strategy> <max_frame_num> <generative_model_id>"
    exit 1
fi
GEN_EXTRA_ARGS=${GEN_EXTRA_ARGS:-}
EXTRA_ARGS=()
if [ -n "$GEN_EXTRA_ARGS" ]; then
    read -r -a EXTRA_ARGS <<< "$GEN_EXTRA_ARGS"
fi
    

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 llava/eval/model_sqa3d.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 8 \
    --question-file data/processed/sqa3d_test_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --overwrite_cfg true \
    --generative_model_id $4 \
    "${EXTRA_ARGS[@]}"
    
python llava/eval/eval_sqa3d.py --input-file $ANWSER_FILE
