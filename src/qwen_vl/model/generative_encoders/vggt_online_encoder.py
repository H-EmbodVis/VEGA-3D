from __future__ import annotations

import json
import math
import os
import sys
import importlib
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import resize_center_crop, resolve_inference_dtype, split_frames, to_unit_range


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


def _resolve_vggt_model_name(raw_name: Optional[str]) -> str:
    if raw_name is None:
        return "facebook/VGGT-1B"
    name = str(raw_name).strip()
    if name == "" or name.lower() == "none":
        return "facebook/VGGT-1B"
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
        raise ImportError(f"Loading `{path}` requires `safetensors` to be installed.") from exc
    return safe_load_file(path, device="cpu")


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if path.is_file():
        if path.suffix == ".safetensors":
            return _load_safetensors(str(path))
        return _load_state_dict_file(str(path))

    # Hugging Face sharded safetensors.
    sharded_index_path = path / "model.safetensors.index.json"
    if sharded_index_path.exists():
        with open(sharded_index_path, "r") as f:
            sharded_index = json.load(f)
        shard_files = sorted(set(sharded_index.get("weight_map", {}).values()))
        if len(shard_files) == 0:
            raise ValueError(f"Invalid sharded safetensors index: {sharded_index_path}")
        merged_state_dict = {}
        for shard_file in shard_files:
            shard_path = path / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing sharded safetensors file: {shard_path}")
            merged_state_dict.update(_load_safetensors(str(shard_path)))
        return merged_state_dict

    preferred_files = (
        "model.safetensors",
        "model.pt",
        "pytorch_model.bin",
        "model.bin",
        "model.pth",
    )
    for file_name in preferred_files:
        candidate = path / file_name
        if candidate.exists():
            return _load_checkpoint_state_dict(str(candidate))

    candidate_files = sorted(
        list(path.glob("*.safetensors"))
        + list(path.glob("*.pt"))
        + list(path.glob("*.bin"))
        + list(path.glob("*.pth"))
    )
    if len(candidate_files) == 0:
        raise FileNotFoundError(
            f"No checkpoint file found in directory: {checkpoint_path}. "
            "Expected model.pt / model.safetensors / pytorch_model.bin / *.pth."
        )
    if len(candidate_files) > 1:
        raise ValueError(
            f"Multiple checkpoint files found in `{checkpoint_path}`. "
            "Please pass an explicit file path via `generative_vision_tower_checkpoint`. "
            f"Candidates: {[str(x) for x in candidate_files]}"
        )
    return _load_checkpoint_state_dict(str(candidate_files[0]))


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
    known_prefixes = ("module.", "model.", "backbone.")
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
            "Could not match VGGT parameters from checkpoint. "
            f"matched_keys={best_overlap}, required>={min_required_overlap}"
        )
    return best_state_dict


def _ensure_vggt_import():
    if importlib.util.find_spec("vggt") is None:
        repo_root = Path(__file__).resolve().parents[3]
        local_vggt_root = repo_root / "extract" / "vggt"
        if local_vggt_root.exists():
            local_vggt_root_str = str(local_vggt_root)
            if local_vggt_root_str not in sys.path:
                sys.path.insert(0, local_vggt_root_str)

    try:
        from vggt.models.vggt import VGGT  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "VGGT import failed. Expected either installed package `vggt` or local source "
            "at `extract/vggt`. Please check your environment."
        ) from exc


class VGGTOnlineEncoder(nn.Module):
    """
    Online VGGT feature encoder.
    Input: [N, 3, H, W], output: [N, Cg, output_spatial, output_spatial].
    """

    def __init__(self, config):
        super().__init__()
        _ensure_vggt_import()
        from vggt.models.vggt import VGGT

        self.model_name = _resolve_vggt_model_name(getattr(config, "generative_vision_tower_model_name", None))
        self.checkpoint_path = getattr(
            config,
            "generative_vision_tower_checkpoint",
            os.getenv("VGGT_MODEL_PATH", None),
        )
        if self.checkpoint_path in ("", "none", "None"):
            self.checkpoint_path = None
        self.pretrained = _safe_bool(getattr(config, "generative_vision_tower_pretrained", False), False)
        default_input_size = _safe_int(getattr(config, "generative_vision_tower_img_size", 518), 518)
        self.input_size = _safe_int(getattr(config, "generative_vision_tower_input_size", default_input_size), default_input_size)
        self.output_spatial = _safe_int(getattr(config, "generative_vision_tower_output_spatial", 14), 14)
        self.chunk_size = _safe_int(getattr(config, "generative_vision_tower_chunk_size", 8), 8)
        self.feat_block_idx = _safe_int(getattr(config, "generative_vision_tower_feat_block_idx", -1), -1)
        if self.input_size <= 0 or (self.input_size % 14 != 0):
            raise ValueError(f"generative_vision_tower_input_size must be positive and divisible by 14, got {self.input_size}")
        if self.chunk_size <= 0:
            raise ValueError(f"generative_vision_tower_chunk_size must be > 0, got {self.chunk_size}")
        self.param_dtype = resolve_inference_dtype(config)
        self._model_device = torch.device("cpu")

        if self.checkpoint_path is None and not self.pretrained:
            raise ValueError(
                "VGGT requires either `generative_vision_tower_checkpoint` (local model.pt) "
                "or `generative_vision_tower_pretrained=True`."
            )

        if self.checkpoint_path is None and self.pretrained:
            self.model = VGGT.from_pretrained(self.model_name).eval()
        else:
            self.model = VGGT().eval()
            self._load_local_checkpoint(self.checkpoint_path)

        self.model.requires_grad_(False)
        # VGGT aggregator output channels = 2 * aggregator embed_dim
        self.embed_dim = int(self.model.aggregator.camera_token.shape[-1]) * 2

    def _load_local_checkpoint(self, checkpoint_path: str):
        raw_state_dict = _load_checkpoint_state_dict(checkpoint_path)
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
        return x

    def _tokens_to_spatial(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise RuntimeError(f"Expected [T, P, C], got {tuple(patch_tokens.shape)}")
        t, token_count, channels = patch_tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise RuntimeError(
                f"Cannot reshape VGGT patch tokens to square map: token_count={token_count}, "
                f"shape={tuple(patch_tokens.shape)}."
            )
        feat = patch_tokens.reshape(t, side, side, channels).permute(0, 3, 1, 2).contiguous()
        if self.output_spatial > 0 and side != self.output_spatial:
            feat = F.adaptive_avg_pool2d(feat, output_size=(self.output_spatial, self.output_spatial))
        return feat

    def _forward_single_video(self, frames: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(frames).to(dtype=self.param_dtype)
        outs = []
        block_idx = self.feat_block_idx if self.feat_block_idx >= 0 else len(self.model.aggregator.frame_blocks) + self.feat_block_idx
        if block_idx < 0:
            raise ValueError(
                f"generative_vision_tower_feat_block_idx out of range: {self.feat_block_idx}. "
                f"Minimum valid value is {-len(self.model.aggregator.frame_blocks)}."
            )
        amp_ctx = (
            torch.autocast(device_type=x.device.type, dtype=self.param_dtype)
            if x.device.type == "cuda" and self.param_dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            for i in range(0, x.shape[0], self.chunk_size):
                x_chunk = x[i : i + self.chunk_size]  # [Tc, 3, H, W]
                images = x_chunk.unsqueeze(0)  # [1, Tc, 3, H, W]
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
                if block_idx < 0 or block_idx >= len(aggregated_tokens_list):
                    raise ValueError(
                        f"generative_vision_tower_feat_block_idx out of range: {self.feat_block_idx}. "
                        f"Valid range: [-{len(aggregated_tokens_list)}, {len(aggregated_tokens_list) - 1}]"
                    )
                tokens = aggregated_tokens_list[block_idx]  # [1, Tc, P, C]
                patch_tokens = tokens[:, :, int(patch_start_idx) :, :].squeeze(0)  # [Tc, Ppatch, C]
                feat = self._tokens_to_spatial(patch_tokens)
                outs.append(feat)
                del aggregated_tokens_list, tokens, patch_tokens
        return torch.cat(outs, dim=0)

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

        self._move_model_to_device(frames.device)
        chunks = split_frames(frames, split_sizes)
        outputs = []
        for chunk in chunks:
            outputs.append(self._forward_single_video(chunk))
        return torch.cat(outputs, dim=0)
