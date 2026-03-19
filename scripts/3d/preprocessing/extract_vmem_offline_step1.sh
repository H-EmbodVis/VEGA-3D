#!/bin/bash
set -euo pipefail

python scripts/3d/preprocessing/extract_seva_vmem_offline_step1_features.py \
  --model_type vmem \
  --annotation_dir data/embodiedscan \
  --video_folder data \
  --save_root data/scannet \
  --save_subdir uniform_vmem_middle_feats_onestep_t250_input16_patch14 \
  --checkpoint_dir data/models/vmem \
  --weight_name vmem_weights.pth \
  --model_version 1.0 \
  --sd_model_path data/models/stable-diffusion-2-1-base \
  --clip_model_path data/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors \
  --num_frames 32 \
  --input_size 896 \
  --output_spatial 14 \
  --timestep 250 \
  --cfg_scale 2.0 \
  --cfg_min 1.2 \
  --camera_scale 2.0 \
  --num_gpus 8 \
  --skip_saved True
