from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import resolve_inference_dtype, split_frames, temporal_resample, to_unit_range
from .svd.svd_vision_tower import SVDVisionBackbone


class SVDOnlineEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.svd_model_path = getattr(
            config,
            "generative_vision_tower_checkpoint",
            os.getenv("SVD_MODEL_PATH", "data/models/stable-video-diffusion-img2vid"),
        )
        self.clip_model_path = getattr(
            config,
            "generative_vision_tower_clip_checkpoint",
            os.getenv("SVD_CLIP_MODEL_PATH", "data/models/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        )
        self.height = int(getattr(config, "generative_vision_tower_height", 896))
        self.width = int(getattr(config, "generative_vision_tower_width", 896))
        self.output_spatial = int(getattr(config, "generative_vision_tower_output_spatial", 14))
        self.chunk_size = int(getattr(config, "generative_vision_tower_chunk_size", 8))
        if self.chunk_size <= 0:
            raise ValueError(f"generative_vision_tower_chunk_size must be > 0, got {self.chunk_size}")
        self.param_dtype = resolve_inference_dtype(config)
        self._model_device = torch.device("cpu")

        self.backbone = SVDVisionBackbone(
            vision_backbone_id="svd",
            image_resize_strategy="resize-naive",
            default_image_size=224,
            height=self.height,
            width=self.width,
            num_frames=8,
            svd_model_path=self.svd_model_path,
            clip_model_path=self.clip_model_path,
            torch_dtype=self.param_dtype,
        ).eval()

    @staticmethod
    def _module_has_meta_tensors(module: Optional[nn.Module]) -> bool:
        if module is None:
            return False
        for p in module.parameters():
            if getattr(p, "is_meta", False):
                return True
        for b in module.buffers():
            if getattr(b, "is_meta", False):
                return True
        return False

    def _rematerialize_backbone_if_needed(self) -> None:
        has_meta = (
            self._module_has_meta_tensors(self.backbone.unet)
            or self._module_has_meta_tensors(self.backbone.image_encoder)
            or self._module_has_meta_tensors(self.backbone.vae)
        )
        if not has_meta:
            return

        # Rebuild once outside HF lazy-init context. This keeps runtime stable
        # when the parent model was loaded via `device_map` and submodules were
        # initialized on `meta`.
        self.backbone.pipeline, self.backbone.vae, self.backbone.unet = self.backbone._load_svd_models()
        self.backbone.image_encoder = self.backbone._load_image_encoder()
        self.backbone.pipeline.image_encoder = self.backbone.image_encoder
        self.backbone.eval()

    def _move_models_to_device(self, device: torch.device):
        if self._model_device == device:
            return

        self._rematerialize_backbone_if_needed()

        # Do not call `pipeline.to(...)` directly: diffusers pipelines may
        # contain optional modules left on `meta` that are unused in this path,
        # which can raise on `.to()`.
        self.backbone.unet.to(device=device, dtype=self.param_dtype)
        self.backbone.image_encoder.to(device=device, dtype=self.param_dtype)
        if self.backbone.vae is not None:
            self.backbone.vae.to(device=device, dtype=self.param_dtype)
        self._model_device = device

    def _forward_single_video(self, frames: torch.Tensor) -> torch.Tensor:
        x = to_unit_range(frames)
        x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=False)
        x = x.to(dtype=self.backbone.torch_dtype)
        chunk_feats = []
        with torch.inference_mode():
            for i in range(0, x.shape[0], self.chunk_size):
                x_chunk = x[i : i + self.chunk_size]
                feats = self.backbone.ddim_one_step(
                    image=x_chunk[:1],
                    pipeline=self.backbone.pipeline,
                    vae=self.backbone.vae,
                    unet=self.backbone.unet,
                    image_encoder=self.backbone.image_encoder,
                    height=self.height,
                    width=self.width,
                    num_frames=x_chunk.shape[0],
                    output_type="unet_latent",
                    all_frames_pixels=x_chunk.unsqueeze(0),
                )
                if not isinstance(feats, torch.Tensor) or feats.ndim != 5:
                    raise RuntimeError(f"Unexpected SVD output: {type(feats)} {getattr(feats, 'shape', None)}")
                feat = feats[0]  # [T', C, H, W]
                if feat.shape[1] > 1280:
                    feat = feat[:, :1280]
                if self.output_spatial > 0 and (feat.shape[-2] != self.output_spatial or feat.shape[-1] != self.output_spatial):
                    feat = F.adaptive_avg_pool2d(feat, output_size=(self.output_spatial, self.output_spatial))
                if feat.shape[0] != x_chunk.shape[0]:
                    feat = temporal_resample(feat, x_chunk.shape[0])
                chunk_feats.append(feat)

        feat = torch.cat(chunk_feats, dim=0)
        if feat.shape[0] != frames.shape[0]:
            feat = temporal_resample(feat, frames.shape[0])
        return feat

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
            return frames.new_zeros((0, 1280, _sp, _sp))
        self._move_models_to_device(frames.device)
        chunks = split_frames(frames, split_sizes)
        outputs = []
        for chunk in chunks:
            outputs.append(self._forward_single_video(chunk))
        return torch.cat(outputs, dim=0)
