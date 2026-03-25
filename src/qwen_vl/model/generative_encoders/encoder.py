"""High-level wrapper around a generative vision tower for Qwen2.5-VL.

The key difference from the geometry-encoder path is *spatial alignment*:
Qwen2.5-VL produces a **dynamic** number of visual tokens per image
(determined by the image resolution after alignment to the 28-pixel grid),
whereas the upstream generative towers output features at their own
**native** spatial resolution.  ``GenerativeVisionEncoder`` solves this
mismatch by:

1. Disabling the fixed-size spatial pooling inside the tower
   (``output_spatial`` is set to ``0``) so that features are returned at
   the tower's native spatial resolution (e.g. WAN-VACE → ``30 × 52``,
   SD2.1 mid-block → ``14 × 14`` at 896 input, VJEPA → ``14 × 14`` at
   224 input, etc.).
2. Using ``F.interpolate`` to resize each image's feature map from the
   tower's native grid to match the Qwen visual encoder's per-image
   patch grid (``h_patches × w_patches``, derived from ``image_grid_thw``).

This avoids the lossy 14×14 bottleneck: the interpolation now starts from
the tower's full native resolution, preserving significantly more spatial
detail.

The interpolated features are then flattened to ``[N, h_patches * w_patches, C]``
and returned for further processing (merger → fusion) in the model class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import build_generative_encoder


@dataclass
class GenerativeEncoderConfig:
    """Lightweight config container forwarded to the upstream tower builder."""

    tower_type: str = "wan_vace_online"
    checkpoint: Optional[str] = None
    freeze: bool = True
    # Extra kwargs forwarded verbatim to the tower constructor via config attrs.
    tower_kwargs: Dict = field(default_factory=dict)


class GenerativeVisionEncoder(nn.Module):
    """Wraps an upstream generative tower with dynamic spatial interpolation.

    After construction the following attributes are available:

    * ``embed_dim``  — number of feature channels produced by the tower.
    * ``tower``      — the raw upstream ``nn.Module``.
    """

    def __init__(self, gen_config: GenerativeEncoderConfig):
        super().__init__()
        self._gen_config = gen_config

        # Build a tiny namespace object that looks like an HF config so the
        # upstream builder's ``getattr(config, ...)`` calls work.
        _cfg = _build_tower_config(gen_config)

        self.tower, self.embed_dim = build_generative_encoder(_cfg)

        if gen_config.freeze:
            self.tower.requires_grad_(False)
            self.tower.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_feature_dim(self) -> int:
        """Return the per-token feature dimension (= channel count)."""
        return self.embed_dim

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Encode raw images and optionally align to Qwen spatial grid.

        Args:
            images: ``[N, 3, H, W]`` raw RGB frames (same as
                ``geometry_encoder_inputs[bn]``).
            grid_thw: ``[N, 3]`` tensor where each row is ``(t, h_patches,
                w_patches)`` for the corresponding image from Qwen's image
                processor.  When provided the output features are
                interpolated to ``h_patches × w_patches`` spatial tokens per
                image.  When ``None`` the raw tower spatial size is kept.

        Returns:
            ``[N, h_patches * w_patches, C]`` if *grid_thw* is given, else
            ``[N, Hs * Ws, C]`` (Hs/Ws = tower's native spatial size).
        """
        self._ensure_tower_materialized()

        # Upstream tower: [N, C, Hs, Ws]
        feat = self.tower(images)

        if grid_thw is not None:
            feat = self._align_to_grid(feat, grid_thw)
        else:
            # Just flatten spatial dims: [N, C, Hs, Ws] → [N, Hs*Ws, C]
            n, c, hs, ws = feat.shape
            feat = feat.reshape(n, c, hs * ws).permute(0, 2, 1).contiguous()

        return feat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_to_grid(
        self, feat: torch.Tensor, grid_thw: torch.LongTensor
    ) -> torch.Tensor:
        """Interpolate each image's feature map to its Qwen patch grid size.

        ``grid_thw[i] = (t, h_patches, w_patches)`` where ``h_patches`` and
        ``w_patches`` are the number of 14-pixel patches along each axis as
        assigned by the Qwen2.5-VL image processor.

        The output is ``[N, h_patches * w_patches, C]``.
        """
        n, c, hs, ws = feat.shape
        assert n == grid_thw.shape[0], (
            f"Feature batch size ({n}) != grid_thw rows ({grid_thw.shape[0]})"
        )

        parts: List[torch.Tensor] = []
        for i in range(n):
            _, h_patch, w_patch = grid_thw[i].tolist()
            h_patch, w_patch = int(h_patch), int(w_patch)

            f_i = feat[i : i + 1]  # [1, C, Hs, Ws]
            if hs != h_patch or ws != w_patch:
                f_i = F.interpolate(
                    f_i.float(),
                    size=(h_patch, w_patch),
                    mode="bilinear",
                    align_corners=False,
                ).to(feat.dtype)
            # [1, C, h_patch, w_patch] → [1, h_patch*w_patch, C]
            f_i = f_i.reshape(1, c, h_patch * w_patch).permute(0, 2, 1)
            parts.append(f_i)

        return torch.cat(parts, dim=0)  # [N, h_patches*w_patches, C]

    def load_model(self, path: str):
        """Load tower weights from *path* (no-op for frozen online towers)."""
        pass

    def _tower_has_meta_params(self) -> bool:
        for p in self.tower.parameters():
            if getattr(p, "is_meta", False):
                return True
        for b in self.tower.buffers():
            if getattr(b, "is_meta", False):
                return True
        return False

    def _ensure_tower_materialized(self) -> None:
        if not self._tower_has_meta_params():
            return

        # If HF `from_pretrained(..., device_map=...)` initialized this module
        # under meta tensors, rebuild tower once outside lazy-init context.
        prev_dim = self.embed_dim
        cfg = _build_tower_config(self._gen_config)
        self.tower, self.embed_dim = build_generative_encoder(cfg)
        if self._gen_config.freeze:
            self.tower.requires_grad_(False)
            self.tower.eval()

        if self.embed_dim != prev_dim:
            raise RuntimeError(
                f"Rebuilt generative tower changed embed_dim from {prev_dim} to {self.embed_dim}."
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _Namespace:
    """Minimal attribute container that mimics an HF config for getattr()."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _build_tower_config(gen_config: GenerativeEncoderConfig) -> _Namespace:
    """Create the pseudo-config expected by the upstream builder."""
    attrs = {
        "generative_vision_tower_type": gen_config.tower_type,
        "generative_encoder_type": gen_config.tower_type,
        # Disable the fixed-size spatial pooling inside the tower so that
        # features are returned at their native spatial resolution.  The
        # wrapper's ``_align_to_grid`` will handle the interpolation to
        # each image's actual Qwen patch grid.
        "generative_vision_tower_output_spatial": 0,
    }
    if gen_config.checkpoint is not None:
        attrs["generative_vision_tower_checkpoint"] = gen_config.checkpoint
    # Forward any extra kwargs the user attached
    attrs.update(gen_config.tower_kwargs)
    return _Namespace(**attrs)
