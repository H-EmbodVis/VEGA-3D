from __future__ import annotations

import os
from contextlib import nullcontext
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from .common import resize_center_crop, resolve_inference_dtype, split_frames, to_neg_one_to_one


class SD21UNet2DConditionModel(UNet2DConditionModel):
    """
    SD2.1 UNet that additionally exposes the pre-midblock activation (`mid_in`)
    for generative feature extraction.
    """

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, None)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        mid_input = sample

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=None,
            )

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if (not is_final_block) and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if getattr(upsample_block, "has_cross_attention", False):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                    upsample_size=upsample_size,
                    attention_mask=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if hasattr(self, "conv_norm_out") and self.conv_norm_out is not None:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        noise_pred = self.conv_out(sample)
        return {"mid_in": mid_input, "noise_pred": noise_pred}


class SD21OnlineEncoder(nn.Module):
    """
    Online SD2.1 generative encoder aligned with offline `extract_sd_features_concurrent.py`:
    - preprocess to 896x896 (default) with center-crop
    - VAE encode -> add noise -> one-step DDIM denoise
    - take UNet pre-midblock feature as generative feature
    """

    def __init__(self, config):
        super().__init__()
        self.checkpoint_dir = getattr(
            config,
            "generative_vision_tower_checkpoint",
            os.getenv("SD21_BASE_CKPT_DIR", "ckpts/stable-diffusion-2-1-base"),
        )
        self.input_size = int(getattr(config, "generative_vision_tower_input_size", 896))
        self.output_spatial = int(getattr(config, "generative_vision_tower_output_spatial", 14))
        self.timestep = int(getattr(config, "generative_vision_tower_timestep", 250))
        self.num_inference_steps = int(getattr(config, "generative_vision_tower_num_inference_steps", 1))
        self.chunk_size = int(getattr(config, "generative_vision_tower_chunk_size", 8))
        self.empty_prompt_path = str(
            getattr(
                config,
                "generative_vision_tower_empty_prompt_path",
                os.getenv("SD21_EMPTY_PROMPT_PATH", os.path.join(self.checkpoint_dir, "empty_prompt_embeds.pt")),
            )
        )
        if self.num_inference_steps <= 0:
            raise ValueError(f"generative_vision_tower_num_inference_steps must be > 0, got {self.num_inference_steps}")
        if self.chunk_size <= 0:
            raise ValueError(f"generative_vision_tower_chunk_size must be > 0, got {self.chunk_size}")

        self.param_dtype = resolve_inference_dtype(config)
        self._model_device = torch.device("cpu")

        self.unet = SD21UNet2DConditionModel.from_pretrained(
            self.checkpoint_dir,
            subfolder="unet",
            force_download=False,
            low_cpu_mem_usage=False,
        ).eval()
        self.unet.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(
            self.checkpoint_dir,
            subfolder="vae",
            force_download=False,
            low_cpu_mem_usage=False,
        ).eval()
        self.vae.requires_grad_(False)
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "force_upcast"):
            self.vae.config.force_upcast = False
        self.vae.decoder = None

        self.scheduler = DDIMScheduler.from_pretrained(
            self.checkpoint_dir,
            subfolder="scheduler",
            force_download=False,
        )

        if not os.path.exists(self.empty_prompt_path):
            raise FileNotFoundError(
                f"Missing empty prompt embedding file: {self.empty_prompt_path}. "
                "Please run: `python scripts/3d/preprocessing/extract_sd21_empty_prompt_embeds.py "
                "--checkpoint_dir <sd21_dir> --output_path <empty_prompt_embeds.pt>`"
            )
        empty_prompt = torch.load(self.empty_prompt_path, map_location="cpu")
        if isinstance(empty_prompt, dict):
            for key in ("prompt_embeds", "empty_prompt_embeds", "embeds"):
                if key in empty_prompt and torch.is_tensor(empty_prompt[key]):
                    empty_prompt = empty_prompt[key]
                    break
        if not torch.is_tensor(empty_prompt):
            raise TypeError(
                f"Unsupported empty prompt format in {self.empty_prompt_path}: {type(empty_prompt)}"
            )
        if empty_prompt.ndim == 2:
            empty_prompt = empty_prompt.unsqueeze(0)
        if empty_prompt.ndim != 3:
            raise ValueError(
                f"Expected empty prompt embedding ndim=3, got shape={tuple(empty_prompt.shape)}"
            )
        cross_dim = int(getattr(self.unet.config, "cross_attention_dim", empty_prompt.shape[-1]))
        if empty_prompt.shape[-1] != cross_dim:
            raise ValueError(
                f"Empty prompt embedding dim mismatch: expected last dim {cross_dim}, got {empty_prompt.shape[-1]}"
            )
        if empty_prompt.shape[0] != 1:
            empty_prompt = empty_prompt[:1]
        self.empty_prompt_embeds = empty_prompt.detach().cpu().to(self.param_dtype).contiguous()

    def _move_models_to_device(self, device: torch.device):
        if self._model_device == device:
            return
        self.vae.to(device=device, dtype=self.param_dtype)
        self.unet.to(device=device, dtype=self.param_dtype)
        self._model_device = device

    def _select_timesteps(self, device: torch.device) -> torch.Tensor:
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        if timesteps.numel() == 0:
            raise ValueError("Scheduler timesteps is empty.")
        target = torch.tensor(int(self.timestep), device=device, dtype=torch.long)
        ts = timesteps.to(torch.long)
        idx = (ts <= target).nonzero(as_tuple=False)
        start_idx = int(idx[0].item()) if idx.numel() > 0 else 0
        timesteps = timesteps[start_idx:]
        if timesteps.numel() == 0:
            timesteps = self.scheduler.timesteps[-1:].clone()
        return timesteps

    def _forward_single_video(self, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = to_neg_one_to_one(frames)
        x = resize_center_crop(x, self.input_size, self.input_size).to(device=device, dtype=self.param_dtype)
        outs = []
        scale_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        timesteps = self._select_timesteps(device=device)
        t_start = timesteps[0]

        with torch.inference_mode():
            for i in range(0, x.shape[0], self.chunk_size):
                x_chunk = x[i : i + self.chunk_size]
                batch_size = x_chunk.shape[0]
                prompt_embeds = self.empty_prompt_embeds.to(device=device, dtype=self.param_dtype)
                prompt_embeds = prompt_embeds.expand(batch_size, -1, -1).contiguous()

                latents = scale_factor * self.vae.encode(x_chunk).latent_dist.mode()
                latents = latents.to(device=device, dtype=self.param_dtype)
                t_start_b = t_start.expand(batch_size)
                noise = torch.randn_like(latents)
                sample = self.scheduler.add_noise(latents, noise, t_start_b).to(device=device, dtype=self.param_dtype)

                mid_input = None
                for t in timesteps:
                    t_b = t.expand(batch_size)
                    out = self.unet(sample, t_b, encoder_hidden_states=prompt_embeds)
                    noise_pred = out["noise_pred"]
                    sample = self.scheduler.step(noise_pred, t, sample).prev_sample
                    mid_input = out["mid_in"]

                if mid_input is None:
                    raise RuntimeError("Failed to capture SD2.1 intermediate feature.")
                feat = mid_input
                if self.output_spatial > 0 and (feat.shape[-2] != self.output_spatial or feat.shape[-1] != self.output_spatial):
                    feat = F.adaptive_avg_pool2d(feat, output_size=(self.output_spatial, self.output_spatial))
                outs.append(feat)

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
            return frames.new_zeros((0, 1280, _sp, _sp))

        device = frames.device
        self._move_models_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outs = []
        autocast_ctx = torch.autocast(device_type=device.type, dtype=self.param_dtype) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            for chunk in chunks:
                outs.append(self._forward_single_video(chunk, device=device))
        return torch.cat(outs, dim=0)
