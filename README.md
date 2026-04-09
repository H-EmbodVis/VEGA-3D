# VEGA-3D Spatial Reasoning

This branch contains the spatial reasoning codebase of **VEGA-3D**. It follows the training and evaluation layout of [VG-LLM](https://github.com/LaVi-Lab/VG-LLM), but adapts it to the VEGA-3D setting built on top of Qwen2.5-VL, geometry encoders, and generative encoders.

This README documents the spatial reasoning workflow only. In particular, it focuses on the `spar_234k + llava_hound_64k` training setup and `VSI-Bench` evaluation. We do **not** include the grounding-oriented pipeline from VG-LLM here.

## What Is Included in This Branch

- Spatial reasoning baselines:
  - `scripts/train/train_sr_qwen3b.sh`
  - `scripts/train/train_sr_vgllm4b.sh`
- VEGA-3D generative-encoder variants under `scripts/train/train_gen_*.sh`
- Recommended open-source release config: `scripts/train/train_gen_wan_t2v_7b.sh`
- VSI-Bench evaluation entrypoint: `scripts/evaluation/eval.sh`
- LMMS-Eval task definition: `src/lmms_eval/tasks/vsibench/`

## Installation

Clone the `spatial-reasoning` branch:

```bash
git clone -b spatial-reasoning https://github.com/H-EmbodVis/VEGA-3D.git
cd VEGA-3D
```

Create the environment:

```bash
conda create -n vega3d-sr python=3.10 -y
conda activate vega3d-sr
pip install --upgrade pip
```

This branch follows the VG-LLM environment layout and was developed with Python 3.10. The released scripts are based on `torch==2.5.1`, `torchvision==0.20.1`, `transformers==4.50.0`, `deepspeed==0.16.4`, and `flash-attn==2.7.4.post1`.

Install PyTorch for your CUDA runtime first, then install the branch dependencies:

```bash
# Install the PyTorch build matching your local CUDA environment.
pip install torch==2.5.1 torchvision==0.20.1

# Core runtime packages used by the released scripts.
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install deepspeed==0.16.4 accelerate==1.4.0 datasets==3.6.0
pip install scipy qwen-vl-utils decord av open3d==0.19.0 timm

# Important: this branch vendors PyTorch3D locally.
# Install the local copy in this repository instead of using a separate external setup.
pip install -e ./pytorch3d

# Install the main package.
pip install -e .
```

Notes:

- Please install the local `./pytorch3d` copy first. This branch already contains the full source tree and does not require a separate external PyTorch3D checkout.
- Several preprocessing and geometry utilities import `scipy` explicitly, so install it even if your environment does not pull it automatically.

## Data and Checkpoints

Following the VG-LLM layout, this branch expects the repository root to contain:

```text
data/
├── models/
├── train/
├── media/
└── evaluation/
```

### 1. Training data

The dataset registry in `src/qwen_vl/data/__init__.py` expects:

- `data/train/spar_234k.json`
- `data/train/llava_hound_64k.json`

Both datasets resolve media files from `data/media`.

This branch follows the same high-level organization as VG-LLM:

- training annotations under `data/train`
- image / video assets under `data/media`
- evaluation datasets under `data/evaluation`

If you already have the raw VLM-3R training data, you can convert it into the branch format with:

```bash
python scripts/preprocess/process_vlm3r_data.py \
  --vlm_3r_data_path /path/to/VLM-3R-DATA \
  --data_source scannet,scannetpp,arkitscenes \
  --frame_num 32 \
  --output_dir data/train
```

The preprocessing script samples multi-view frames from `data/media/scannet`, `data/media/scannetpp`, and `data/media/arkitscenes`, so please keep the media layout consistent with the VG-LLM data preparation.

### 2. VSI-Bench evaluation data

For the workflow documented in this README, the only evaluation dataset we cover is `VSI-Bench`.

The LMMS-Eval task file `src/lmms_eval/tasks/vsibench/vsibench.yaml` points to the Hugging Face dataset `nyu-visionx/VSI-Bench`. A local download layout that matches `scripts/evaluation/eval.sh` is:

```bash
mkdir -p data/evaluation

huggingface-cli download nyu-visionx/VSI-Bench \
  --repo-type dataset \
  --local-dir data/evaluation/VSI-Bench
```

### 3. Model checkpoints

The released scripts use the following local names:

| Checkpoint | Expected local path | Used by |
| --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | `data/models/Qwen2.5-VL-3B-Instruct/` | `scripts/train/train_sr_qwen3b.sh` |
| Qwen2.5-VL-7B-Instruct | `data/models/Qwen2.5-VL-7B-Instruct/` | `scripts/train/train_gen_wan_t2v_7b.sh` and other `train_gen_*_7b.sh` scripts |
| VGGT-1B | `data/models/VGGT-1B/` | `scripts/train/train_sr_vgllm4b.sh` |
| Wan2.1-T2V-1.3B | `data/models/Wan2.1-T2V-1.3B` | `scripts/train/train_gen_wan_t2v_7b.sh` |

Other generative-encoder variants follow the same pattern under `scripts/train/train_gen_*.sh`. Please check the corresponding script for the exact checkpoint directory name before launching training.

## Training

All released spatial reasoning training scripts:

- use `spar_234k,llava_hound_64k`
- auto-detect the number of local GPUs with `nvidia-smi`
- launch training with `torchrun`

### 1. Qwen2.5-VL 3B baseline

```bash
bash scripts/train/train_sr_qwen3b.sh
```

### 2. VGLLM / VGGT-style 4B baseline

```bash
bash scripts/train/train_sr_vgllm4b.sh
```

### 3. Recommended VEGA-3D release setting

```bash
bash scripts/train/train_gen_wan_t2v_7b.sh
```

This script is the recommended open-source release configuration in this branch: Qwen2.5-VL 7B + Wan2.1-T2V-1.3B with token-gated residual fusion.

## Evaluation on VSI-Bench

This README only documents the `VSI-Bench` workflow.

The simplest evaluation path is the provided wrapper:

```bash
export LMMS_EVAL_DATA_ROOT=data/evaluation
export LMMS_EVAL_VSIBENCH_DATASET_PATH=${LMMS_EVAL_DATA_ROOT}/VSI-Bench

bash scripts/evaluation/eval.sh
```

By default, `scripts/evaluation/eval.sh` evaluates the released checkpoint:

```text
zd11024/vgllm-qa-vggt-4b
```

If you want to evaluate a local checkpoint instead, use the same command pattern as the training scripts:

```bash
accelerate launch --num_processes=8 -m lmms_eval \
  --model vgllm \
  --model_args pretrained=ckpt/gen-wan_t2v-7b-token-gated-residual-s1,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
  --tasks vsibench \
  --batch_size 1 \
  --output_path logs/$(date +%Y%m%d)
```

Notes:

- `VSI-Bench` evaluation uses `max_num_frames=32` in the released scripts.
- Several training scripts still keep an additional `CV-Bench` block after the `VSI-Bench` block. If you only want the `VSI-Bench` workflow documented here, run `scripts/evaluation/eval.sh` directly or keep only the first evaluation block in your training script.

## Acknowledgement

This branch builds on the following projects:

- [VG-LLM](https://github.com/LaVi-Lab/VG-LLM)
- [Qwen2.5-VL](https://huggingface.co/Qwen)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

## Citation

If this repository is useful for your research, please consider citing:

```bibtex
@article{wu2026vega,
  title={Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding},
  author={Xianjin Wu and Dingkang Liang and Tianrui Feng and Kui Xia and Yumeng Zhang and Xiaofan Li and Xiao Tan and Xiang Bai},
  journal={arXiv preprint arXiv:2603.19235},
  year={2026}
}
```
