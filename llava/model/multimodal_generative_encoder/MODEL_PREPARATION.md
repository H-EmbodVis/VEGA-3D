# Model Preparation Details

This document collects the extra checkpoint preparation steps referenced by the main README.

## General Rule

Most checkpoints can be downloaded with:

```bash
huggingface-cli download <repo_id> --local-dir <expected_local_path>
```

## DINOv3

The training script expects the local path:

```text
data/models/vit_large_patch16_dinov3.sat493m
```

Download the `timm/vit_large_patch16_dinov3.lvd1689m` repository into that exact directory.

## Stable Diffusion 2.1

`train_sd21_online.sh` expects:

```text
data/models/stable-diffusion-2-1-base/empty_prompt_embeds.pt
```

Generate it after downloading SD2.1:

```bash
python3 scripts/3d/preprocessing/extract_sd21_empty_prompt_embeds.py \
  --checkpoint_dir data/models/stable-diffusion-2-1-base \
  --output_path data/models/stable-diffusion-2-1-base/empty_prompt_embeds.pt
```

## WAN Prompt Embeddings

WAN-based settings additionally require:

```text
llava/model/multimodal_generative_encoder/wan_prompt_embedding.pt
```

Export it with the matching task:

```bash
python3 scripts/3d/preprocessing/export_wan_prompt_embedding.py \
  --task t2v-1.3B \
  --checkpoint_dir data/models/Wan2.1-T2V-1.3B \
  --output llava/model/multimodal_generative_encoder/wan_prompt_embedding.pt

python3 scripts/3d/preprocessing/export_wan_prompt_embedding.py \
  --task vace-1.3B \
  --checkpoint_dir data/models/Wan2.1-VACE-1.3B \
  --output llava/model/multimodal_generative_encoder/wan_prompt_embedding.pt
```

## CLIP For SVD, SEVA, And Vmem

If you use SVD, SEVA, or Vmem, also download:

```bash
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --local-dir data/models/CLIP-ViT-H-14-laion2B-s32B-b79K
```

SEVA/Vmem feature extraction additionally depends on:

```text
data/models/stable-diffusion-2-1-base
data/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors
```

## VGGT

`train_vggt_online.sh` checks for:

```text
data/models/VGGT-1B/model.pt
```

Download the checkpoint into `data/models/VGGT-1B`, and make sure the main weight file is available at `data/models/VGGT-1B/model.pt`.
