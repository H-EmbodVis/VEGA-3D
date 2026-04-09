from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SPLIT = "train"
DEFAULT_ANNOTATION_DIR = "data/embodiedscan"
DEFAULT_VIDEO_FOLDER = "data"
DEFAULT_CHECKPOINT_DIR = "data/models/Wan2.1-T2V-1.3B"
DEFAULT_TASK = "t2v-1.3B"
DEFAULT_SIZE = "832*480"
DEFAULT_NUM_FRAMES = 32
DEFAULT_BLOCK_IDX = 20
DEFAULT_TIMESTEP = 300
DEFAULT_SHIFT = 5.0
DEFAULT_DTYPE = "bf16"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_CHUNK_SIZE = 8
DEFAULT_SEED = 42
DEFAULT_VALID_RATIO_THRESHOLD = 0.1


def normalize_scene_id(scene_id: str) -> str:
    scene_id = str(scene_id).strip()
    if scene_id.startswith("scannet/"):
        return scene_id
    low = scene_id.lower()
    if low.startswith("scene"):
        return f"scannet/{low}"
    if low.startswith("scannet_scene"):
        return f"scannet/{low.split('scannet_')[-1]}"
    return f"scannet/{low}"


def scene_slug(scene_id: str) -> str:
    return normalize_scene_id(scene_id).split("/", 1)[-1]


def default_output_dir(name: str, scene_id: str, object_id: int | None = None) -> str:
    slug = scene_slug(scene_id)
    if object_id is None:
        return f"results/visualization/{name}/{slug}"
    return f"results/visualization/{name}/{slug}_obj{int(object_id)}"


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    common = parser.add_argument_group("Common")
    common.add_argument("--scene_id", type=str, default="scannet/scene0000_00")
    common.add_argument("--output_dir", type=str, default=None, help="Defaults to results/visualization/<task>/<scene>.")
    common.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    common.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    common.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    common.add_argument("--block_idx", type=int, default=DEFAULT_BLOCK_IDX)
    common.add_argument("--timestep", type=int, default=DEFAULT_TIMESTEP)

    advanced = parser.add_argument_group("Advanced Runtime")
    advanced.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["train", "val", "test"])
    advanced.add_argument("--annotation_dir", type=str, default=DEFAULT_ANNOTATION_DIR)
    advanced.add_argument("--video_folder", type=str, default=DEFAULT_VIDEO_FOLDER)
    advanced.add_argument("--size", type=str, default=DEFAULT_SIZE)
    advanced.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["bf16", "fp16", "fp32"])
    advanced.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    advanced.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def build_runtime_namespace(args, output_dir: str, **extra) -> argparse.Namespace:
    payload = {
        "scene_id": normalize_scene_id(args.scene_id),
        "split": args.split,
        "annotation_dir": args.annotation_dir,
        "video_folder": args.video_folder,
        "checkpoint_dir": args.checkpoint_dir,
        "task": DEFAULT_TASK,
        "size": args.size,
        "num_frames": int(args.num_frames),
        "feat_block_idx": int(args.block_idx),
        "timestep": int(args.timestep),
        "shift": DEFAULT_SHIFT,
        "dtype": args.dtype,
        "device": args.device,
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "output_dir": output_dir,
    }
    payload.update(extra)
    return argparse.Namespace(**payload)
