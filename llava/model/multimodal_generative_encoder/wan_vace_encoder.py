import math
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import resize_center_crop, resolve_inference_dtype, split_frames, to_neg_one_to_one
from .wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from .wan.modules.vae import WanVAE
from .wan.modules.vace_model import VaceWanModel
from .wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanVaceOnlineEncoder(nn.Module):
    """
    Online WAN-VACE feature encoder.
    Input: [N, 3, H, W], output: [N, Cg, 14, 14].
    """

    def __init__(self, config):
        super().__init__()
        self.task = getattr(config, "generative_vision_tower_task", getattr(config, "generative_encoder_task", "vace-1.3B"))
        self.checkpoint_dir = getattr(config, "generative_vision_tower_checkpoint", getattr(config, "generative_encoder_checkpoint", ""))
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.getenv("WAN_VACE_CKPT_DIR", "")
        if not self.checkpoint_dir:
            raise ValueError(
                "Online generative vision tower requires `generative_vision_tower_checkpoint` or `WAN_VACE_CKPT_DIR`."
            )
        self.size = getattr(config, "generative_vision_tower_size", getattr(config, "generative_encoder_size", "832*480"))
        self.timestep = int(getattr(config, "generative_vision_tower_timestep", getattr(config, "generative_encoder_timestep", 300)))
        self.shift = float(getattr(config, "generative_vision_tower_shift", getattr(config, "generative_encoder_shift", 5.0)))
        self.context_scale = float(getattr(config, "generative_vision_tower_context_scale", getattr(config, "generative_encoder_context_scale", 1.0)))
        self.feat_block_idx = int(getattr(config, "generative_vision_tower_feat_block_idx", getattr(config, "generative_encoder_feat_block_idx", -1)))
        self.condition_on_first_frame = bool(getattr(config, "generative_vision_tower_condition_on_first_frame", getattr(config, "generative_encoder_condition_on_first_frame", True)))
        self.prompt_emb_path = str(
            getattr(
                config,
                "generative_vision_tower_prompt_emb_path",
                os.getenv("WAN_PROMPT_EMBED_PATH", os.path.join(os.path.dirname(__file__), "wan_prompt_embedding.pt")),
            )
        )

        if self.task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported WAN task: {self.task}")
        if self.size not in SIZE_CONFIGS:
            raise ValueError(f"Unsupported WAN size: {self.size}")

        self.cfg = WAN_CONFIGS[self.task]
        self.param_dtype = resolve_inference_dtype(config)
        self.num_train_timesteps = self.cfg.num_train_timesteps
        self.vae_stride = self.cfg.vae_stride
        self.patch_size = self.cfg.patch_size
        self.frame_width, self.frame_height = SIZE_CONFIGS[self.size]
        self._model_device = torch.device("cpu")

        self.vae = WanVAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.cfg.vae_checkpoint),
            dtype=self.param_dtype,
            device=torch.device("cpu"),
        )
        if not os.path.exists(self.prompt_emb_path):
            raise FileNotFoundError(
                f"Prompt embedding not found: {self.prompt_emb_path}. "
                "Please run the offline extraction script first."
            )
        prompt_context = torch.load(self.prompt_emb_path, map_location="cpu", weights_only=True)
        if isinstance(prompt_context, dict):
            if "context" in prompt_context:
                prompt_context = prompt_context["context"]
            elif "embedding" in prompt_context:
                prompt_context = prompt_context["embedding"]
        if not torch.is_tensor(prompt_context):
            raise ValueError(f"Invalid prompt embedding file: {self.prompt_emb_path}")
        self.prompt_context = prompt_context.detach().cpu().contiguous()

        self.model = VaceWanModel.from_pretrained(self.checkpoint_dir).eval().requires_grad_(False)
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self._scheduler_device = None

    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Convert input frames to [-1, 1], resize/crop to WAN resolution.
        """
        x = to_neg_one_to_one(frames)
        x = resize_center_crop(x, self.frame_height, self.frame_width)
        return x.to(dtype=self.param_dtype)

    def _get_text_context(self, device: torch.device, batch_size: int):
        context = self.prompt_context.to(device=device, non_blocking=True)
        if torch.is_floating_point(context):
            context = context.to(self.param_dtype)
        return [context] * batch_size

    def _ensure_scheduler_ready(self, device: torch.device):
        if self._scheduler_device != device:
            # Use full train-time schedule to preserve exact timestep semantics.
            self.scheduler.set_timesteps(self.num_train_timesteps, device=device, shift=self.shift)
            self._scheduler_device = device

    def _select_timestep(self, timesteps: torch.Tensor, target_timestep: int) -> torch.Tensor:
        if timesteps.numel() == 0:
            raise ValueError("Scheduler timesteps is empty.")
        # We do one-step denoise at the target timestep used for noise injection.
        # With full schedule, this is effectively an exact match.
        tau_tensor = torch.tensor(int(target_timestep), device=timesteps.device, dtype=timesteps.dtype)
        idx = torch.argmin(torch.abs(timesteps - tau_tensor))
        return timesteps[idx]

    def _move_vae_to_device(self, device: torch.device):
        if self.vae.device != device:
            self.vae.model.to(device)
            self.vae.mean = self.vae.mean.to(device=device)
            self.vae.std = self.vae.std.to(device=device)
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
            self.vae.device = device

    def _move_models_to_device(self, device: torch.device):
        if self._model_device != device:
            self.model.to(device)
            self._move_vae_to_device(device)
            self._model_device = device

    @staticmethod
    def _vace_latent(z: List[torch.Tensor], m: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def _vace_encode_frames(self, frames: List[torch.Tensor], ref_images=None, masks=None):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive)
            reactive = self.vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                ref_latent = self.vae.encode(refs)
                if masks is not None:
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def _vace_encode_masks(self, masks: List[torch.Tensor], ref_images=None) -> List[torch.Tensor]:
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0])
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            mask = mask[0, :, :, :]
            mask = mask.view(depth, height, self.vae_stride[1], width, self.vae_stride[1])
            mask = mask.permute(2, 4, 0, 1, 3)
            mask = mask.reshape(self.vae_stride[1] * self.vae_stride[2], depth, height, width)
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(new_depth, height, width),
                mode="nearest-exact",
            ).squeeze(0)
            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def _forward_single_video(self, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            return frames.new_zeros((0, getattr(self.cfg, "dim", 1280), 14, 14))

        x = self._prepare_frames(frames)
        frame_list = [x[i].unsqueeze(1) for i in range(x.shape[0])]  # [C, 1, H, W]
        masks = [torch.ones((1, 1, self.frame_height, self.frame_width), device=device) for _ in range(len(frame_list))]
        ref_frame = frame_list[0] if self.condition_on_first_frame else None

        self._ensure_scheduler_ready(device)
        tau = self._select_timestep(self.scheduler.timesteps, target_timestep=self.timestep)
        context = self._get_text_context(device=device, batch_size=len(frame_list))

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.param_dtype):
            base_latents = self.vae.encode(frame_list)
            if ref_frame is None:
                ref_images = [None] * len(frame_list)
            else:
                ref_images = [[ref_frame] for _ in range(len(frame_list))]
            z0 = self._vace_encode_frames(frame_list, ref_images=ref_images, masks=masks)
            m0 = self._vace_encode_masks(masks, ref_images=ref_images)
            z = self._vace_latent(z0, m0)

            target_shape = list(z0[0].shape)
            target_shape[0] = int(target_shape[0] / 2)
            seq_len = math.ceil(
                (target_shape[2] * target_shape[3]) / (self.patch_size[1] * self.patch_size[2]) * target_shape[1]
            )

            latent_batch = torch.stack(base_latents, dim=0)
            noise = torch.randn_like(latent_batch)
            noisy_latents = self.scheduler.add_noise(
                original_samples=latent_batch,
                noise=noise,
                timesteps=tau.expand(len(frame_list)),
            )
            noisy_latents_list = [noisy_latents[i] for i in range(len(frame_list))]

            feat_holder = {}

            def _hook(_, __, output):
                feat_holder["feat"] = output.detach()

            block_idx = (len(self.model.blocks) - 1) if self.feat_block_idx < 0 else self.feat_block_idx
            if block_idx < 0 or block_idx >= len(self.model.blocks):
                raise ValueError(f"feat_block_idx out of range: {block_idx}")

            handle = self.model.blocks[block_idx].register_forward_hook(_hook)
            out_batch = None
            try:
                t = tau.expand(len(frame_list)).to(device=device, dtype=torch.long)
                out_list = self.model(
                    noisy_latents_list,
                    t=t,
                    vace_context=z,
                    vace_context_scale=self.context_scale,
                    context=context,
                    seq_len=seq_len,
                )
                # Feature extraction uses one denoiser forward at target timestep.
                # No iterative/multistep scheduler update is needed here.
                out_batch = torch.stack(out_list, dim=0)
            finally:
                handle.remove()

            if "feat" not in feat_holder:
                raise RuntimeError("Failed to capture WAN-VACE intermediate features.")

            feats = feat_holder["feat"]  # [N, L, C]
            grid_h = self.frame_height // (self.vae_stride[1] * self.patch_size[1])
            grid_w = self.frame_width // (self.vae_stride[2] * self.patch_size[2])
            tokens_per_frame = grid_h * grid_w
            if feats.shape[1] == tokens_per_frame * 2:
                feats = feats[:, tokens_per_frame:, :]
            elif feats.shape[1] != tokens_per_frame:
                raise RuntimeError(
                    f"Unexpected token count: {feats.shape[1]} (expected {tokens_per_frame} or {tokens_per_frame * 2})."
                )

            feats = feats.view(feats.shape[0], grid_h, grid_w, feats.shape[2]).permute(0, 3, 1, 2).contiguous()
            feats = F.adaptive_avg_pool2d(feats, output_size=(14, 14))
            # Release large temporaries early to reduce peak memory.
            del latent_batch, noisy_latents, noisy_latents_list, noise
            if out_batch is not None:
                del out_batch
            return feats

    def forward(
        self,
        frames: torch.Tensor,
        split_sizes: List[int] | None = None,
        video_contexts=None,
    ) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            return frames.new_zeros((0, getattr(self.cfg, "dim", 1280), 14, 14))

        device = frames.device
        self._move_models_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outs = []
        for chunk in chunks:
            outs.append(self._forward_single_video(chunk, device=device))
        return torch.cat(outs, dim=0)
