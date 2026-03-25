"""Generative vision encoders for VG-LLM.

Wraps multimodal_generative_encoder towers (SD2.1, WAN-VACE, SVD, V-JEPA,
DINOv3, VGGT-online, VAE) and provides a unified interface for the
Qwen2.5-VL integration pipeline.
"""

from .encoder import GenerativeVisionEncoder
from .builder import build_generative_encoder, TOWER_EMBED_DIMS

__all__ = [
    "GenerativeVisionEncoder",
    "build_generative_encoder",
    "TOWER_EMBED_DIMS",
]
