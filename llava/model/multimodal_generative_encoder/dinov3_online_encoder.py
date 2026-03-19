from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import resize_center_crop, resolve_inference_dtype, split_frames, to_unit_range

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DINO_V3_MODEL_ALIASES = {
    "small": "vit_small_patch16_dinov3",
    "small_plus": "vit_small_plus_patch16_dinov3",
    "base": "vit_base_patch16_dinov3",
    "large": "vit_large_patch16_dinov3",
    "huge_plus": "vit_huge_plus_patch16_dinov3",
    "7b": "vit_7b_patch16_dinov3",
    "small_qkvb": "vit_small_patch16_dinov3_qkvb",
    "small_plus_qkvb": "vit_small_plus_patch16_dinov3_qkvb",
    "base_qkvb": "vit_base_patch16_dinov3_qkvb",
    "large_qkvb": "vit_large_patch16_dinov3_qkvb",
    "huge_plus_qkvb": "vit_huge_plus_patch16_dinov3_qkvb",
}


def _safe_int(value, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "none":
            return int(default)
    return int(value)


def _safe_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
        if v == "none":
            return bool(default)
    return bool(value)


def _resolve_model_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if name in DINO_V3_MODEL_ALIASES:
        return DINO_V3_MODEL_ALIASES[name]
    return name


def _load_state_dict_file(path: str) -> Dict:
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


def _discover_checkpoint_file(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if path.is_file():
        return path

    preferred_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "model.pt",
        "model.pth",
    )
    for file_name in preferred_files:
        candidate = path / file_name
        if candidate.exists():
            return candidate

    candidate_files = sorted(
        list(path.glob("*.safetensors"))
        + list(path.glob("*.bin"))
        + list(path.glob("*.pt"))
        + list(path.glob("*.pth"))
    )
    if len(candidate_files) == 0:
        raise FileNotFoundError(
            f"No checkpoint file found in directory: {checkpoint_path}. "
            "Expected model.safetensors / pytorch_model.bin / *.pt."
        )
    if len(candidate_files) > 1:
        raise ValueError(
            f"Multiple checkpoint files found in `{checkpoint_path}`. "
            "Please pass an explicit file path via `generative_vision_tower_checkpoint`. "
            f"Candidates: {[str(x) for x in candidate_files]}"
        )
    return candidate_files[0]


def _iter_candidate_state_dicts(raw_state_dict: Dict):
    if not isinstance(raw_state_dict, dict):
        return
    queue = [raw_state_dict]
    visited = set()
    candidate_keys = (
        "state_dict",
        "model_state_dict",
        "model",
        "module",
        "backbone",
        "encoder",
        "teacher",
        "student",
    )
    while len(queue) > 0:
        cur = queue.pop(0)
        if not isinstance(cur, dict):
            continue
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)
        yield cur
        for key in candidate_keys:
            next_item = cur.get(key, None)
            if isinstance(next_item, dict):
                queue.append(next_item)


def _strip_prefixes(key: str) -> str:
    known_prefixes = ("module.", "model.", "backbone.", "encoder.", "teacher.", "student.")
    cleaned = key
    changed = True
    while changed:
        changed = False
        for prefix in known_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                changed = True
    return cleaned


def _select_best_state_dict(raw_state_dict: Dict, model_keys: set) -> Dict:
    best_overlap = -1
    best_state_dict = None
    for candidate_state_dict in _iter_candidate_state_dicts(raw_state_dict):
        normalized = {}
        for key, value in candidate_state_dict.items():
            if torch.is_tensor(value):
                normalized[_strip_prefixes(key)] = value
        overlap = sum(1 for key in normalized.keys() if key in model_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state_dict = normalized

    min_required_overlap = max(16, int(0.03 * len(model_keys)))
    if best_state_dict is None or best_overlap < min_required_overlap:
        raise ValueError(
            "Could not match DINOv3 parameters from checkpoint. "
            f"matched_keys={best_overlap}, required>={min_required_overlap}"
        )
    return best_state_dict


class DINOv3OnlineEncoder(nn.Module):
    """
    Online DINOv3 feature encoder.
    Input: [N, 3, H, W], output: [N, Cg, output_spatial, output_spatial].
    """

    def __init__(self, config):
        super().__init__()
        raw_model_name = getattr(config, "generative_vision_tower_model_name", "vit_large_patch16_dinov3")
        self.model_name = _resolve_model_name(raw_model_name)
        self.checkpoint_path = getattr(
            config,
            "generative_vision_tower_checkpoint",
            os.getenv("DINOV3_MODEL_PATH", None),
        )
        if self.checkpoint_path in ("", "none", "None"):
            self.checkpoint_path = None
        self.pretrained = _safe_bool(getattr(config, "generative_vision_tower_pretrained", True), True)
        default_input_size = _safe_int(getattr(config, "generative_vision_tower_img_size", 224), 224)
        self.input_size = _safe_int(getattr(config, "generative_vision_tower_input_size", default_input_size), default_input_size)
        self.output_spatial = _safe_int(getattr(config, "generative_vision_tower_output_spatial", 14), 14)
        self.chunk_size = _safe_int(getattr(config, "generative_vision_tower_chunk_size", 32), 32)
        if self.chunk_size <= 0:
            raise ValueError(f"generative_vision_tower_chunk_size must be > 0, got {self.chunk_size}")
        self.param_dtype = resolve_inference_dtype(config)
        self._model_device = torch.device("cpu")

        if self.model_name not in timm.list_models("*dinov3*"):
            raise ValueError(
                f"Unsupported DINOv3 model `{self.model_name}`. "
                "Please use a timm DINOv3 model name, e.g. vit_small/base/large/hp_patch16_dinov3."
            )

        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained and self.checkpoint_path is None,
                num_classes=0,
                global_pool="",
            ).eval()
        except Exception as exc:
            msg = str(exc)
            if self.pretrained and self.checkpoint_path is None and (
                "OfflineModeIsEnabled" in msg or "LocalEntryNotFoundError" in msg
            ):
                raise RuntimeError(
                    "DINOv3 pretrained weights are requested but cannot be downloaded in offline mode.\n"
                    "Please provide local weights via `--generative_vision_tower_checkpoint <path>` "
                    "and set `--generative_vision_tower_pretrained False`, "
                    "or disable offline mode for this run."
                ) from exc
            raise
        self.model.requires_grad_(False)
        if self.checkpoint_path is not None:
            self._load_local_checkpoint(self.checkpoint_path)
            self.model.eval()
            self.model.requires_grad_(False)

        self.embed_dim = int(getattr(self.model, "num_features", getattr(self.model, "embed_dim", 1024)))
        patch_attr = getattr(getattr(self.model, "patch_embed", None), "patch_size", 16)
        self.patch_size = int(patch_attr[0] if isinstance(patch_attr, (tuple, list)) else patch_attr)
        self.tokens_per_frame = (self.input_size // self.patch_size) * (self.input_size // self.patch_size)
        self.mean, self.std = self._resolve_normalization_stats()

    def _resolve_normalization_stats(self):
        try:
            from timm.data import resolve_model_data_config

            data_cfg = resolve_model_data_config(self.model)
            mean = tuple(data_cfg.get("mean", IMAGENET_DEFAULT_MEAN))
            std = tuple(data_cfg.get("std", IMAGENET_DEFAULT_STD))
            return mean, std
        except Exception:
            return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    def _load_local_checkpoint(self, checkpoint_path: str):
        checkpoint_file = _discover_checkpoint_file(checkpoint_path)
        if checkpoint_file.suffix == ".safetensors":
            raw_state_dict = _load_safetensors(str(checkpoint_file))
        else:
            raw_state_dict = _load_state_dict_file(str(checkpoint_file))
        model_keys = set(self.model.state_dict().keys())
        state_dict = _select_best_state_dict(raw_state_dict, model_keys=model_keys)
        self.model.load_state_dict(state_dict, strict=False)

    def _move_model_to_device(self, device: torch.device):
        if self._model_device != device:
            self.model.to(device=device, dtype=self.param_dtype)
            self._model_device = device

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        x = to_unit_range(frames)
        x = resize_center_crop(x, self.input_size, self.input_size)
        mean = x.new_tensor(self.mean)[None, :, None, None]
        std = x.new_tensor(self.std)[None, :, None, None]
        x = (x - mean) / std
        return x

    def _extract_patch_tokens(self, feats) -> torch.Tensor:
        if isinstance(feats, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state", "x"):
                value = feats.get(key, None)
                if torch.is_tensor(value):
                    feats = value
                    break
            else:
                tensor_values = [value for value in feats.values() if torch.is_tensor(value)]
                if len(tensor_values) == 0:
                    raise RuntimeError(f"Could not find tensor output in DINOv3 forward_features dict: keys={list(feats.keys())}")
                feats = tensor_values[0]

        if not torch.is_tensor(feats):
            raise RuntimeError(f"Unexpected DINOv3 forward_features output type: {type(feats)}")
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)
        if feats.ndim != 3:
            raise RuntimeError(f"Expected DINOv3 tokens [B, L, C], got shape={tuple(feats.shape)}")

        expected_patches = int(self.tokens_per_frame)
        if feats.shape[1] >= expected_patches:
            if feats.shape[1] > expected_patches:
                feats = feats[:, -expected_patches:, :]
            return feats

        # Fallback: infer patch tokens from output length when dynamic resolution handling differs.
        total_tokens = int(feats.shape[1])
        for prefix_tokens in (0, 1, 4, 5, 8):
            patch_tokens = total_tokens - prefix_tokens
            if patch_tokens <= 0:
                continue
            side = int(math.sqrt(patch_tokens))
            if side * side == patch_tokens:
                return feats[:, prefix_tokens:, :]

        raise RuntimeError(
            f"Could not infer DINOv3 patch tokens from output shape {tuple(feats.shape)} "
            f"(expected at least {expected_patches} patch tokens)."
        )

    def _tokens_to_spatial(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        bsz, token_count, channels = patch_tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise RuntimeError(
                f"Cannot reshape DINOv3 tokens to square map: token_count={token_count}, shape={tuple(patch_tokens.shape)}."
            )
        feat = patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side).contiguous()
        if side != self.output_spatial:
            feat = F.adaptive_avg_pool2d(feat, output_size=(self.output_spatial, self.output_spatial))
        return feat

    def _forward_single_video(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(frames).to(dtype=self.param_dtype)
        outputs = []
        amp_ctx = (
            torch.autocast(device_type=x.device.type, dtype=self.param_dtype)
            if x.device.type == "cuda" and self.param_dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            for i in range(0, x.shape[0], self.chunk_size):
                x_chunk = x[i : i + self.chunk_size]
                feats = self.model.forward_features(x_chunk)
                patch_tokens = self._extract_patch_tokens(feats)
                outputs.append(self._tokens_to_spatial(patch_tokens))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        frames: torch.Tensor,
        split_sizes: Optional[List[int]] = None,
        video_contexts=None,
    ) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            return frames.new_zeros((0, self.embed_dim, self.output_spatial, self.output_spatial))

        device = frames.device
        self._move_model_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outputs = []
        for chunk in chunks:
            outputs.append(self._forward_single_video(chunk))
        return torch.cat(outputs, dim=0)
