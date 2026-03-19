import argparse
from typing import Dict, Optional

import torch
from transformers import AutoConfig


GENERIC_OVERWRITE_BASE = {
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 151649,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def add_generative_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument("--use_generative_feature", type=str2bool, default=None)
    parser.add_argument("--generative_feat_dim", type=int, default=None)
    parser.add_argument("--generative_projector_type", type=str, default=None)
    parser.add_argument("--generative_feature_source", type=str, default="auto", choices=["auto", "offline", "online", "none"])
    parser.add_argument("--generative_vision_tower_type", type=str, default=None)
    parser.add_argument("--generative_vision_tower_checkpoint", type=str, default=None)
    parser.add_argument("--generative_vision_tower_task", type=str, default=None)
    parser.add_argument("--generative_vision_tower_size", type=str, default=None)
    parser.add_argument("--generative_vision_tower_timestep", type=int, default=None)
    parser.add_argument("--generative_vision_tower_shift", type=float, default=None)
    parser.add_argument("--generative_vision_tower_context_scale", type=float, default=None)
    parser.add_argument("--generative_vision_tower_feat_block_idx", type=int, default=None)
    parser.add_argument("--generative_vision_tower_condition_on_first_frame", type=str2bool, default=None)
    parser.add_argument("--generative_vision_tower_weight_name", type=str, default=None)
    parser.add_argument("--generative_vision_tower_model_version", type=float, default=None)
    parser.add_argument("--generative_vision_tower_input_size", type=int, default=None)
    parser.add_argument("--generative_vision_tower_output_spatial", type=int, default=None)
    parser.add_argument("--generative_vision_tower_height", type=int, default=None)
    parser.add_argument("--generative_vision_tower_width", type=int, default=None)
    parser.add_argument("--generative_vision_tower_num_frames", type=int, default=None)
    parser.add_argument("--generative_vision_tower_num_inference_steps", type=int, default=None)
    parser.add_argument("--generative_vision_tower_pool_kernel", type=int, default=None)
    parser.add_argument("--generative_vision_tower_model_name", type=str, default=None)
    parser.add_argument("--generative_vision_tower_img_size", type=int, default=None)
    parser.add_argument("--generative_vision_tower_tubelet_size", type=int, default=None)
    parser.add_argument("--generative_vision_tower_pretrained", type=str2bool, default=None)
    parser.add_argument("--generative_vision_tower_clip_checkpoint", type=str, default=None)
    parser.add_argument("--generative_vision_tower_sd21_checkpoint", type=str, default=None)
    parser.add_argument("--generative_vision_tower_chunk_size", type=int, default=None)
    parser.add_argument("--generative_vision_tower_cfg_min", type=float, default=None)
    parser.add_argument("--generative_vision_tower_camera_scale", type=float, default=None)
    parser.add_argument("--generative_cache_max_items", type=int, default=None)
    parser.add_argument("--generative_profile", type=str2bool, default=None)


def _extract_runtime_overrides(args) -> Dict:
    keys = [
        "use_generative_feature",
        "generative_feat_dim",
        "generative_projector_type",
        "generative_vision_tower_type",
        "generative_vision_tower_checkpoint",
        "generative_vision_tower_task",
        "generative_vision_tower_size",
        "generative_vision_tower_timestep",
        "generative_vision_tower_shift",
        "generative_vision_tower_context_scale",
        "generative_vision_tower_feat_block_idx",
        "generative_vision_tower_condition_on_first_frame",
        "generative_vision_tower_weight_name",
        "generative_vision_tower_model_version",
        "generative_vision_tower_input_size",
        "generative_vision_tower_output_spatial",
        "generative_vision_tower_height",
        "generative_vision_tower_width",
        "generative_vision_tower_num_frames",
        "generative_vision_tower_num_inference_steps",
        "generative_vision_tower_pool_kernel",
        "generative_vision_tower_model_name",
        "generative_vision_tower_img_size",
        "generative_vision_tower_tubelet_size",
        "generative_vision_tower_pretrained",
        "generative_vision_tower_clip_checkpoint",
        "generative_vision_tower_sd21_checkpoint",
        "generative_vision_tower_chunk_size",
        "generative_vision_tower_cfg_min",
        "generative_vision_tower_camera_scale",
        "generative_cache_max_items",
        "generative_profile",
    ]

    cfg = {}
    for k in keys:
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None:
                cfg[k] = v

    if hasattr(args, "generative_feature_source") and getattr(args, "generative_feature_source", "auto") != "auto":
        cfg["generative_feature_source"] = args.generative_feature_source

    return cfg


def build_model_overwrite_config(args) -> Optional[Dict]:
    cfg = {}
    if getattr(args, "lora_path", None) is not None:
        cfg = AutoConfig.from_pretrained(args.lora_path).to_dict()
    elif bool(getattr(args, "overwrite_cfg", False)):
        cfg.update(GENERIC_OVERWRITE_BASE)

    cfg.update(_extract_runtime_overrides(args))
    return cfg if len(cfg) > 0 else None


def resolve_generative_feature_source(model_config, args) -> str:
    if hasattr(args, "generative_feature_source") and args.generative_feature_source != "auto":
        return args.generative_feature_source

    use_generative_feature = bool(getattr(model_config, "use_generative_feature", False))
    if not use_generative_feature:
        return "none"
    return getattr(model_config, "generative_feature_source", "offline")


def move_video_dict_to_device(video_dict: Dict, device, floating_dtype: torch.dtype = torch.bfloat16):
    if "images" not in video_dict:
        raise KeyError("`images` is required in video_dict")

    image_tensors = video_dict.pop("images")
    if torch.is_floating_point(image_tensors):
        image_tensors = image_tensors.to(device=device, dtype=floating_dtype)
    else:
        image_tensors = image_tensors.to(device=device)

    for key, value in list(video_dict.items()):
        if not torch.is_tensor(value):
            continue
        if torch.is_floating_point(value):
            video_dict[key] = value.to(device=device, dtype=floating_dtype)
        else:
            video_dict[key] = value.to(device=device)

    return image_tensors, video_dict
