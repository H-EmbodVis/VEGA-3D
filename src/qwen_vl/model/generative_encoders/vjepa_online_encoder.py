from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import resolve_inference_dtype, split_frames, temporal_resample, to_unit_range
from .vjepa.hub import backbones as vjepa_backbones

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
VJEPA_MODEL_NAME_ALIASES = {
    "vitg": "vit_giant",
    "vitg-384": "vit_giant_384",
    "vit_giant_xformers": "vit_giant",
    "vitg_fpc64_384": "vit_giant_384",
    "vitl": "vit_large",
    "vith": "vit_huge",
}


def _resolve_vjepa_model_name(raw_name: Optional[str], checkpoint_path: Optional[str]) -> str:
    if raw_name is None:
        name = ""
    else:
        name = str(raw_name).strip()

    if name == "" or name.lower() == "none":
        hint = str(checkpoint_path or "").lower()
        if "dinov3" in hint:
            return "vit_giant"
        if "ac" in hint:
            return "vit_ac_giant"
        if "384" in hint or "vitg-384" in hint:
            return "vit_giant_384"
        if "vitl" in hint or "large" in hint:
            return "vit_large"
        if "vith" in hint or "huge" in hint:
            return "vit_huge"
        return "vit_giant"

    lowered = name.lower()
    if lowered in VJEPA_MODEL_NAME_ALIASES:
        return VJEPA_MODEL_NAME_ALIASES[lowered]

    # Common accidental carry-over from DINO configuration.
    if "dinov3" in lowered:
        return "vit_giant"

    return name


def _load_state_dict(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_safetensors(path: str) -> Dict:
    try:
        from safetensors.torch import load_file as safe_load_file
    except Exception as exc:
        raise ImportError(
            f"Loading `{path}` requires `safetensors` to be installed."
        ) from exc
    return safe_load_file(path, device="cpu")


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict:
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if ckpt_path.is_file():
        if ckpt_path.suffix == ".safetensors":
            return _load_safetensors(str(ckpt_path))
        return _load_state_dict(str(ckpt_path))

    # Hugging Face sharded safetensors
    sharded_index_path = ckpt_path / "model.safetensors.index.json"
    if sharded_index_path.exists():
        with open(sharded_index_path, "r") as f:
            sharded_index = json.load(f)
        shard_files = sorted(set(sharded_index.get("weight_map", {}).values()))
        if len(shard_files) == 0:
            raise ValueError(f"Invalid sharded safetensors index: {sharded_index_path}")
        merged_state_dict = {}
        for shard_file in shard_files:
            shard_path = ckpt_path / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing sharded safetensors file: {shard_path}")
            merged_state_dict.update(_load_safetensors(str(shard_path)))
        return merged_state_dict

    preferred_names = ("model.safetensors", "pytorch_model.bin", "model.bin", "model.pt", "model.pth")
    for preferred_name in preferred_names:
        preferred_path = ckpt_path / preferred_name
        if preferred_path.exists():
            return _load_checkpoint_state_dict(str(preferred_path))

    candidate_files = sorted(
        list(ckpt_path.glob("*.safetensors"))
        + list(ckpt_path.glob("*.bin"))
        + list(ckpt_path.glob("*.pt"))
        + list(ckpt_path.glob("*.pth"))
    )
    if len(candidate_files) == 0:
        raise FileNotFoundError(
            f"No checkpoint file found in directory: {checkpoint_path}. "
            "Expected one of model.safetensors / pytorch_model.bin / *.pt."
        )
    if len(candidate_files) > 1:
        raise ValueError(
            f"Multiple checkpoint files found in `{checkpoint_path}`; "
            "please pass an explicit file path via `generative_vision_tower_checkpoint`. "
            f"Candidates: {[str(x) for x in candidate_files]}"
        )
    return _load_checkpoint_state_dict(str(candidate_files[0]))


def _iter_candidate_state_dicts(state_dict: Dict):
    queue = [state_dict]
    visited = set()
    candidate_keys = ("state_dict", "model_state_dict", "model", "module", "encoder", "backbone")

    while len(queue) > 0:
        cur = queue.pop(0)
        if not isinstance(cur, dict):
            continue
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)
        yield cur
        for candidate_key in candidate_keys:
            next_item = cur.get(candidate_key, None)
            if isinstance(next_item, dict):
                queue.append(next_item)


def _strip_known_prefixes(key: str) -> str:
    known_prefixes = (
        "module.",
        "model.",
        "encoder.",
        "backbone.",
        "vision_model.",
        "vjepa2.",
    )
    cleaned_key = key
    changed = True
    while changed:
        changed = False
        for known_prefix in known_prefixes:
            if cleaned_key.startswith(known_prefix):
                cleaned_key = cleaned_key[len(known_prefix) :]
                changed = True
    return cleaned_key


def _convert_hf_vjepa2_encoder_state_dict(state_dict: Dict) -> Optional[Dict]:
    """
    Convert HF VJEPA2 encoder-style keys to the vendored VJEPA backbone keys.
    HF example:
      vjepa2.encoder.layer.0.attention.query.weight
    Vendored example:
      blocks.0.attn.qkv.weight
    """
    if not any(k.startswith("vjepa2.encoder.layer.") for k in state_dict.keys()):
        return None

    converted = {}

    # Patch embedding
    patch_weight = state_dict.get("vjepa2.encoder.embeddings.patch_embeddings.proj.weight", None)
    patch_bias = state_dict.get("vjepa2.encoder.embeddings.patch_embeddings.proj.bias", None)
    if torch.is_tensor(patch_weight):
        converted["patch_embed.proj.weight"] = patch_weight
    if torch.is_tensor(patch_bias):
        converted["patch_embed.proj.bias"] = patch_bias

    # Collect layer ids from keys like: vjepa2.encoder.layer.<idx>.*
    layer_id_pattern = re.compile(r"^vjepa2\.encoder\.layer\.(\d+)\.")
    layer_ids = set()
    for key in state_dict.keys():
        match = layer_id_pattern.match(key)
        if match is not None:
            layer_ids.add(int(match.group(1)))

    for layer_id in sorted(layer_ids):
        prefix = f"vjepa2.encoder.layer.{layer_id}"
        out_prefix = f"blocks.{layer_id}"

        # Attention qkv: HF keeps q/k/v separate; vendored model uses merged qkv.
        for wb in ("weight", "bias"):
            q = state_dict.get(f"{prefix}.attention.query.{wb}", None)
            k = state_dict.get(f"{prefix}.attention.key.{wb}", None)
            v = state_dict.get(f"{prefix}.attention.value.{wb}", None)
            if torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v):
                converted[f"{out_prefix}.attn.qkv.{wb}"] = torch.cat([q, k, v], dim=0)

        # Attention output projection
        for wb in ("weight", "bias"):
            attn_proj = state_dict.get(f"{prefix}.attention.proj.{wb}", None)
            if torch.is_tensor(attn_proj):
                converted[f"{out_prefix}.attn.proj.{wb}"] = attn_proj

        # Layer norms
        for wb in ("weight", "bias"):
            norm1 = state_dict.get(f"{prefix}.norm1.{wb}", None)
            norm2 = state_dict.get(f"{prefix}.norm2.{wb}", None)
            if torch.is_tensor(norm1):
                converted[f"{out_prefix}.norm1.{wb}"] = norm1
            if torch.is_tensor(norm2):
                converted[f"{out_prefix}.norm2.{wb}"] = norm2

        # MLP
        for wb in ("weight", "bias"):
            fc1 = state_dict.get(f"{prefix}.mlp.fc1.{wb}", None)
            fc2 = state_dict.get(f"{prefix}.mlp.fc2.{wb}", None)
            if torch.is_tensor(fc1):
                converted[f"{out_prefix}.mlp.fc1.{wb}"] = fc1
            if torch.is_tensor(fc2):
                converted[f"{out_prefix}.mlp.fc2.{wb}"] = fc2

    # Final encoder norm
    for wb in ("weight", "bias"):
        encoder_norm = state_dict.get(f"vjepa2.encoder.layernorm.{wb}", None)
        if torch.is_tensor(encoder_norm):
            converted[f"norm.{wb}"] = encoder_norm

    if len(converted) == 0:
        return None
    return converted


def _load_pretrained_vjepa_weights(model, checkpoint_path: str):
    raw_state_dict = _load_checkpoint_state_dict(checkpoint_path)
    model_keys = set(model.state_dict().keys())

    best_overlap = -1
    best_state_dict = None

    for candidate_state_dict in _iter_candidate_state_dicts(raw_state_dict):
        variants = []

        normalized_state_dict = {}
        for key, value in candidate_state_dict.items():
            if torch.is_tensor(value):
                normalized_state_dict[_strip_known_prefixes(key)] = value
        variants.append(normalized_state_dict)

        converted_hf_state_dict = _convert_hf_vjepa2_encoder_state_dict(candidate_state_dict)
        if converted_hf_state_dict is not None:
            variants.append(converted_hf_state_dict)

        for state_dict_variant in variants:
            overlap = sum(1 for key in state_dict_variant.keys() if key in model_keys)
            if overlap > best_overlap:
                best_overlap = overlap
                best_state_dict = state_dict_variant

    min_required_overlap = max(32, int(0.05 * len(model_keys)))
    if best_state_dict is None or best_overlap < min_required_overlap:
        raise ValueError(
            "Could not match VJEPA encoder parameters from checkpoint. "
            f"checkpoint_path={checkpoint_path}, matched_keys={best_overlap}, "
            f"required>={min_required_overlap}"
        )

    model.load_state_dict(best_state_dict, strict=False)


class VJEPAOnlineEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        raw_model_name = getattr(config, "generative_vision_tower_model_name", None)
        self.img_size = int(getattr(config, "generative_vision_tower_img_size", 224))
        self.tubelet_size = int(getattr(config, "generative_vision_tower_tubelet_size", 2))
        self.num_frames = int(getattr(config, "generative_vision_tower_num_frames", 64))
        self.pretrained = bool(getattr(config, "generative_vision_tower_pretrained", False))
        self.checkpoint_path = getattr(
            config,
            "generative_vision_tower_checkpoint",
            os.getenv("VJEPA2_MODEL_DIR", os.getenv("VJEPA_PT_MODEL_PATH", None)),
        )
        if self.checkpoint_path in ("", "none", "None"):
            self.checkpoint_path = None
        self.model_name = _resolve_vjepa_model_name(raw_model_name, self.checkpoint_path)
        if self.model_name not in vjepa_backbones.ARCH_NAME_MAP:
            raise ValueError(
                f"Unsupported VJEPA model_name `{self.model_name}`. "
                f"Supported values: {sorted(vjepa_backbones.ARCH_NAME_MAP.keys())}"
            )
        if self.checkpoint_path is None and not self.pretrained:
            raise ValueError(
                "VJEPA online encoder requires either `generative_vision_tower_checkpoint` "
                "(checkpoint file or Hugging Face local model directory) "
                "or `generative_vision_tower_pretrained=True`."
            )
        self.output_spatial = int(getattr(config, "generative_vision_tower_output_spatial", 14))
        self.param_dtype = resolve_inference_dtype(config)
        self._model_device = torch.device("cpu")

        encoder, _ = vjepa_backbones._make_vjepa2_model(
            model_name=self.model_name,
            img_size=self.img_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            pretrained=self.pretrained and self.checkpoint_path is None,
        )
        if self.checkpoint_path is not None:
            _load_pretrained_vjepa_weights(encoder, self.checkpoint_path)
        self.model = encoder.eval()
        self.model.requires_grad_(False)
        self.embed_dim = int(getattr(self.model, "embed_dim", 1408))

    def _move_model_to_device(self, device: torch.device):
        if self._model_device != device:
            self.model.to(device=device, dtype=self.param_dtype)
            self._model_device = device

    @staticmethod
    def _resize_short_side(video: torch.Tensor, short_side: int) -> torch.Tensor:
        _, _, h, w = video.shape
        if min(h, w) == short_side:
            return video
        if h < w:
            new_h = short_side
            new_w = int(round(w * (short_side / h)))
        else:
            new_w = short_side
            new_h = int(round(h * (short_side / w)))
        return F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        x = to_unit_range(frames)
        short_side = int(256.0 / 224.0 * self.img_size)
        x = self._resize_short_side(x, short_side)
        _, _, h, w = x.shape
        top = max(0, (h - self.img_size) // 2)
        left = max(0, (w - self.img_size) // 2)
        x = x[:, :, top : top + self.img_size, left : left + self.img_size]
        mean = x.new_tensor(IMAGENET_DEFAULT_MEAN)[None, :, None, None]
        std = x.new_tensor(IMAGENET_DEFAULT_STD)[None, :, None, None]
        x = (x - mean) / std
        return x

    def _forward_single_video(self, frames: torch.Tensor) -> torch.Tensor:
        orig_frames = int(frames.shape[0])
        x = self._preprocess(frames).to(dtype=self.param_dtype)
        target_frames = min(orig_frames, self.num_frames)
        if x.shape[0] != target_frames:
            x = temporal_resample(x, target_frames)
        if target_frames % self.tubelet_size != 0:
            target_encoder_frames = target_frames + (self.tubelet_size - target_frames % self.tubelet_size)
            x = temporal_resample(x, target_encoder_frames)
        else:
            target_encoder_frames = target_frames

        video = x.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        feats = self.model(video)  # [1, L, C]
        patch_attr = self.model.patch_size
        patch_size = int(patch_attr[0] if isinstance(patch_attr, (tuple, list)) else patch_attr)
        t_patches = target_encoder_frames // self.tubelet_size
        h_patches = self.img_size // patch_size
        w_patches = self.img_size // patch_size
        expected_tokens = t_patches * h_patches * w_patches
        if feats.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Unexpected token count: got {feats.shape[1]}, expected {expected_tokens} "
                f"(T={t_patches}, H={h_patches}, W={w_patches})."
            )
        feats = feats.view(1, t_patches, h_patches, w_patches, feats.shape[-1])
        feats = feats.permute(0, 1, 4, 2, 3).contiguous()[0]  # [T', C, H, W]
        if self.output_spatial > 0 and (feats.shape[-2] != self.output_spatial or feats.shape[-1] != self.output_spatial):
            feats = F.adaptive_avg_pool2d(feats, output_size=(self.output_spatial, self.output_spatial))
        if feats.shape[0] != target_frames:
            feats = temporal_resample(feats, target_frames)
        if target_frames != orig_frames:
            feats = temporal_resample(feats, orig_frames)
        return feats

    def forward(
        self,
        frames: torch.Tensor,
        split_sizes: Optional[List[int]] = None,
        video_contexts=None,
    ) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            _sp = self.output_spatial if self.output_spatial > 0 else 1
            return frames.new_zeros((0, self.embed_dim, _sp, _sp))

        device = frames.device
        self._move_model_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outputs = []
        with torch.inference_mode():
            for chunk in chunks:
                outputs.append(self._forward_single_video(chunk))
        return torch.cat(outputs, dim=0)
