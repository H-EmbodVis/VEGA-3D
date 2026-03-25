"""Builder / factory for generative vision towers.

All encoder implementations live in this package (copied from the upstream
``multimodal_generative_encoder/`` directory).  No external imports are needed.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

# Default embed_dim per tower type (used when the tower instance does not
# expose an ``embed_dim`` attribute).
TOWER_EMBED_DIMS = {
    "wan_vace_online": 1536,
    "wan_t2v_online": 1536,
    "wan_online": 1536,
    "sd21_online": 1280,
    "sd2.1_online": 1280,
    "diffusion_online": 1280,
    "svd_online": 1280,
    "vjepa_online": 1408,
    "jepa_online": 1408,
    "dinov3_online": 1024,
    "dino_online": 1024,
    "vggt_online": 2048,
    "vggt": 2048,
    "vae_online": 4,
}


def _build_tower(config) -> nn.Module:
    """Instantiate the requested generative vision tower."""
    tower_type = getattr(
        config,
        "generative_vision_tower_type",
        getattr(config, "generative_encoder_type", "wan_vace_online"),
    )

    if tower_type in {"wan_vace_online", "wan_t2v_online", "wan_online"}:
        from .wan_vace_encoder import WanVaceOnlineEncoder
        return WanVaceOnlineEncoder(config)

    if tower_type in {"sd21_online", "sd2.1_online", "diffusion_online"}:
        from .sd21_online_encoder import SD21OnlineEncoder
        return SD21OnlineEncoder(config)

    if tower_type == "svd_online":
        from .svd_online_encoder import SVDOnlineEncoder
        return SVDOnlineEncoder(config)

    if tower_type in {"vjepa_online", "jepa_online"}:
        from .vjepa_online_encoder import VJEPAOnlineEncoder
        return VJEPAOnlineEncoder(config)

    if tower_type in {"dinov3_online", "dino_online"}:
        from .dinov3_online_encoder import DINOv3OnlineEncoder
        return DINOv3OnlineEncoder(config)

    if tower_type in {"vggt_online", "vggt"}:
        from .vggt_online_encoder import VGGTOnlineEncoder
        return VGGTOnlineEncoder(config)

    if tower_type == "vae_online":
        from .vae_online_encoder import VAEOnlineEncoder
        return VAEOnlineEncoder(config)

    raise ValueError(f"Unknown generative vision tower: {tower_type}")


def build_generative_encoder(config) -> Tuple[nn.Module, int]:
    """Build a generative tower and return ``(tower, embed_dim)``.

    ``embed_dim`` is the number of feature channels (``C``) in the tower's
    ``[N, C, Hs, Ws]`` output tensor.
    """
    tower = _build_tower(config)
    tower_type = getattr(
        config,
        "generative_vision_tower_type",
        getattr(config, "generative_encoder_type", "wan_vace_online"),
    )
    embed_dim = int(
        getattr(tower, "embed_dim", TOWER_EMBED_DIMS.get(tower_type, 1280))
    )
    return tower, embed_dim
