from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from .common import resize_center_crop, resolve_inference_dtype, split_frames, to_neg_one_to_one


class VAEOnlineEncoder(nn.Module):
    """
    Online VAE feature encoder aligned with offline extraction:
    - RGB frames -> SD2.1 VAE latents
    - optional avg-pooling
    """

    def __init__(self, config):
        super().__init__()
        self.checkpoint_dir = getattr(
            config,
            "generative_vision_tower_checkpoint",
            os.getenv("SD21_BASE_CKPT_DIR", "ckpts/stable-diffusion-2-1-base"),
        )
        self.input_size = int(getattr(config, "generative_vision_tower_input_size", 224))
        self.pool_kernel = int(getattr(config, "generative_vision_tower_pool_kernel", 2))
        self.output_spatial = int(getattr(config, "generative_vision_tower_output_spatial", 14))
        self.chunk_size = int(getattr(config, "generative_vision_tower_chunk_size", 32))
        self.param_dtype = resolve_inference_dtype(config)
        self._model_device = torch.device("cpu")

        self.vae = AutoencoderKL.from_pretrained(
            self.checkpoint_dir,
            subfolder="vae",
            force_download=False,
            low_cpu_mem_usage=False,
        ).eval()
        self.vae.requires_grad_(False)

    def _move_model_to_device(self, device: torch.device):
        if self._model_device != device:
            self.vae.to(device=device, dtype=self.param_dtype)
            self._model_device = device

    def _encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(batch).latent_dist.mean
        latents = latents * 0.18215
        if self.pool_kernel > 1:
            latents = F.avg_pool2d(latents, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        if self.output_spatial > 0 and (latents.shape[-2] != self.output_spatial or latents.shape[-1] != self.output_spatial):
            latents = F.adaptive_avg_pool2d(latents, output_size=(self.output_spatial, self.output_spatial))
        return latents

    def _forward_single_video(self, frames: torch.Tensor) -> torch.Tensor:
        x = to_neg_one_to_one(frames)
        x = resize_center_crop(x, self.input_size, self.input_size).to(dtype=self.param_dtype)
        out = []
        for i in range(0, x.shape[0], max(1, self.chunk_size)):
            out.append(self._encode_batch(x[i : i + self.chunk_size]))
        return torch.cat(out, dim=0)

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
            return frames.new_zeros((0, 4, _sp, _sp))

        device = frames.device
        self._move_model_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outputs = []
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.param_dtype):
            for chunk in chunks:
                outputs.append(self._forward_single_video(chunk))
        return torch.cat(outputs, dim=0)
