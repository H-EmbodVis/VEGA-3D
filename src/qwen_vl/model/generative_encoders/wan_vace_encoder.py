import json
import logging
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import disable_hf_zero3_init, resize_center_crop, resolve_inference_dtype, split_frames, to_neg_one_to_one
from .wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from .wan.modules.model import WanModel
from .wan.modules.vae import WanVAE
from .wan.modules.vace_model import VaceWanModel
from .wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


class WanVaceOnlineEncoder(nn.Module):
    """
    Online WAN feature encoder.
    Supports both Wan2.1 VACE and Wan2.1 T2V checkpoints.
    Input: [N, 3, H, W], output: [N, Cg, grid_h, grid_w] at native spatial resolution.
    """

    def __init__(self, config):
        super().__init__()

        explicit_task = None
        if hasattr(config, "generative_vision_tower_task"):
            explicit_task = getattr(config, "generative_vision_tower_task")
        elif hasattr(config, "generative_encoder_task"):
            explicit_task = getattr(config, "generative_encoder_task")

        self.checkpoint_dir = getattr(config, "generative_vision_tower_checkpoint", getattr(config, "generative_encoder_checkpoint", ""))
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.getenv("WAN_CKPT_DIR", "")
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.getenv("WAN_VACE_CKPT_DIR", "")
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.getenv("WAN_T2V_CKPT_DIR", "")
        if not self.checkpoint_dir:
            raise ValueError(
                "Online generative vision tower requires `generative_vision_tower_checkpoint` "
                "or one of env vars: WAN_CKPT_DIR / WAN_VACE_CKPT_DIR / WAN_T2V_CKPT_DIR."
            )

        self.task = self._resolve_wan_task(explicit_task)
        self._is_vace = self.task.startswith("vace-")
        self._dit_model_cls = VaceWanModel if self._is_vace else WanModel

        self.size = getattr(config, "generative_vision_tower_size", getattr(config, "generative_encoder_size", "832*480"))
        self.timestep = int(getattr(config, "generative_vision_tower_timestep", getattr(config, "generative_encoder_timestep", 300)))
        self.shift = float(getattr(config, "generative_vision_tower_shift", getattr(config, "generative_encoder_shift", 5.0)))
        self.context_scale = float(getattr(config, "generative_vision_tower_context_scale", getattr(config, "generative_encoder_context_scale", 1.0)))
        self.feat_block_idx = int(getattr(config, "generative_vision_tower_feat_block_idx", getattr(config, "generative_encoder_feat_block_idx", -1)))
        self.condition_on_first_frame = bool(getattr(config, "generative_vision_tower_condition_on_first_frame", getattr(config, "generative_encoder_condition_on_first_frame", True)))
        self.prompt_emb_path = getattr(
            config,
            "generative_vision_tower_prompt_emb_path",
            os.path.join(os.path.dirname(__file__), "wan_prompt_embedding.pt"),
        )

        if self.task not in WAN_CONFIGS:
            raise ValueError(f"Unsupported WAN task: {self.task}")
        if self.size not in SIZE_CONFIGS:
            raise ValueError(f"Unsupported WAN size: {self.size}")

        self.cfg = WAN_CONFIGS[self.task]
        self.embed_dim = self.cfg.dim
        self.param_dtype = resolve_inference_dtype(config)
        self.num_train_timesteps = self.cfg.num_train_timesteps
        self.vae_stride = self.cfg.vae_stride
        self.patch_size = self.cfg.patch_size
        self.frame_width, self.frame_height = SIZE_CONFIGS[self.size]
        self._model_device = torch.device("cpu")

        with disable_hf_zero3_init():
            self.vae = WanVAE(
                vae_pth=os.path.join(self.checkpoint_dir, self.cfg.vae_checkpoint),
                dtype=self.param_dtype,
                device=torch.device("cpu"),
            )

        if not self.prompt_emb_path:
            self.prompt_emb_path = os.path.join(os.path.dirname(__file__), "wan_prompt_embedding.pt")
        if not os.path.exists(self.prompt_emb_path):
            ckpt_prompt = os.path.join(self.checkpoint_dir, "wan_prompt_embedding.pt")
            if os.path.exists(ckpt_prompt):
                self.prompt_emb_path = ckpt_prompt
        if not os.path.exists(self.prompt_emb_path):
            raise FileNotFoundError(
                f"WAN prompt embedding not found: {self.prompt_emb_path}. "
                "Please provide `generative_vision_tower_prompt_emb_path` or place "
                "`wan_prompt_embedding.pt` next to this file / in the checkpoint directory."
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

        self.model = self._load_dit_model()
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self._scheduler_device = None

    @staticmethod
    def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if value.lower() in {"", "none", "null", "auto"}:
            return None
        return value

    @staticmethod
    def _detect_wan_family(raw_cfg: dict) -> Optional[str]:
        model_type = str(raw_cfg.get("model_type", "")).strip().lower()
        class_name = str(raw_cfg.get("_class_name", "")).strip().lower()
        if model_type in {"vace", "t2v"}:
            return model_type
        if "vace" in class_name:
            return "vace"
        if class_name == "wanmodel":
            return "t2v"
        return None

    @staticmethod
    def _detect_wan_size(raw_cfg: dict) -> str:
        def _safe_int(key: str) -> Optional[int]:
            value = raw_cfg.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        dim = _safe_int("dim")
        num_layers = _safe_int("num_layers")
        num_heads = _safe_int("num_heads")
        if (
            (dim is not None and dim >= 4096)
            or (num_layers is not None and num_layers >= 40)
            or (num_heads is not None and num_heads >= 40)
        ):
            return "14B"
        return "1.3B"

    @classmethod
    def _detect_task_from_checkpoint(cls, checkpoint_dir: str) -> Optional[str]:
        cfg_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.isfile(cfg_path):
            return None

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
        except Exception as exc:
            logger.warning("Failed to parse WAN config %s: %s", cfg_path, exc)
            return None

        family = cls._detect_wan_family(raw_cfg)
        if family is None:
            return None
        size = cls._detect_wan_size(raw_cfg)

        for task_name in (f"{family}-{size}", f"{family}-1.3B", f"{family}-14B"):
            if task_name in WAN_CONFIGS:
                return task_name
        return None

    def _resolve_wan_task(self, explicit_task: Optional[str]) -> str:
        explicit_task = self._normalize_optional_str(explicit_task)
        detected_task = self._detect_task_from_checkpoint(self.checkpoint_dir)

        if explicit_task is not None:
            if explicit_task not in WAN_CONFIGS:
                raise ValueError(f"Unsupported WAN task: {explicit_task}")
            if detected_task is not None and detected_task != explicit_task:
                logger.warning(
                    "Explicit WAN task `%s` overrides checkpoint-detected task `%s` from %s.",
                    explicit_task,
                    detected_task,
                    self.checkpoint_dir,
                )
            return explicit_task

        if detected_task is not None:
            return detected_task
        return "vace-1.3B"

    def _load_dit_model(self) -> nn.Module:
        with disable_hf_zero3_init():
            return self._dit_model_cls.from_pretrained(self.checkpoint_dir).eval().requires_grad_(False)

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

    def _select_timestep(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.numel() == 0:
            raise ValueError("Scheduler timesteps is empty.")
        # We do one-step denoise at the target timestep used for noise injection.
        # With full schedule, this is effectively an exact match.
        tau_tensor = torch.tensor(int(self.timestep), device=timesteps.device, dtype=timesteps.dtype)
        idx = torch.argmin(torch.abs(timesteps - tau_tensor))
        return timesteps[idx]

    def _move_vae_to_device(self, device: torch.device):
        if self.vae.device != device:
            self.vae.to(device)

    @staticmethod
    def _module_has_meta_tensors(module: nn.Module) -> bool:
        return any(getattr(p, "is_meta", False) for p in module.parameters()) or any(
            getattr(b, "is_meta", False) for b in module.buffers()
        )

    def _rematerialize_if_needed(self):
        if self._module_has_meta_tensors(self.model):
            self.model = self._load_dit_model()
        self.vae.rematerialize_if_needed()

    def _move_models_to_device(self, device: torch.device):
        if self._model_device != device:
            self._rematerialize_if_needed()
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

    def _compute_seq_len(self, latent: torch.Tensor) -> int:
        # latent shape: [C, F, H, W]
        return math.ceil(
            (latent.shape[2] * latent.shape[3]) / (self.patch_size[1] * self.patch_size[2]) * latent.shape[1]
        )

    def _forward_single_video(self, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            _dim = getattr(self.cfg, "dim", 1280)
            _gh = self.frame_height // (self.vae_stride[1] * self.patch_size[1])
            _gw = self.frame_width // (self.vae_stride[2] * self.patch_size[2])
            return frames.new_zeros((0, _dim, _gh, _gw))

        x = self._prepare_frames(frames)
        frame_list = [x[i].unsqueeze(1) for i in range(x.shape[0])]  # [C, 1, H, W]

        self._ensure_scheduler_ready(device)
        tau = self._select_timestep(self.scheduler.timesteps)
        context = self._get_text_context(device=device, batch_size=len(frame_list))

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=self.param_dtype):
            base_latents = self.vae.encode(frame_list)

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

                if self._is_vace:
                    masks = [
                        torch.ones((1, 1, self.frame_height, self.frame_width), device=device)
                        for _ in range(len(frame_list))
                    ]
                    ref_frame = frame_list[0] if self.condition_on_first_frame else None
                    if ref_frame is None:
                        ref_images = [None] * len(frame_list)
                    else:
                        ref_images = [[ref_frame] for _ in range(len(frame_list))]
                    z0 = self._vace_encode_frames(frame_list, ref_images=ref_images, masks=masks)
                    m0 = self._vace_encode_masks(masks, ref_images=ref_images)
                    vace_context = self._vace_latent(z0, m0)

                    target_shape = list(z0[0].shape)
                    target_shape[0] = int(target_shape[0] / 2)
                    seq_len = math.ceil(
                        (target_shape[2] * target_shape[3]) / (self.patch_size[1] * self.patch_size[2]) * target_shape[1]
                    )

                    out_list = self.model(
                        noisy_latents_list,
                        t=t,
                        vace_context=vace_context,
                        vace_context_scale=self.context_scale,
                        context=context,
                        seq_len=seq_len,
                    )
                else:
                    seq_len = self._compute_seq_len(base_latents[0])
                    out_list = self.model(
                        noisy_latents_list,
                        t=t,
                        context=context,
                        seq_len=seq_len,
                    )
                # Feature extraction uses one denoiser forward at target timestep.
                # No iterative/multistep scheduler update is needed here.
                out_batch = torch.stack(out_list, dim=0)
            finally:
                handle.remove()

            if "feat" not in feat_holder:
                raise RuntimeError("Failed to capture WAN intermediate features.")

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
            _dim = getattr(self.cfg, "dim", 1280)
            _gh = self.frame_height // (self.vae_stride[1] * self.patch_size[1])
            _gw = self.frame_width // (self.vae_stride[2] * self.patch_size[2])
            return frames.new_zeros((0, _dim, _gh, _gw))

        device = frames.device
        self._move_models_to_device(device)
        chunks = split_frames(frames, split_sizes)
        outs = []
        for chunk in chunks:
            outs.append(self._forward_single_video(chunk, device=device))
        return torch.cat(outs, dim=0)
