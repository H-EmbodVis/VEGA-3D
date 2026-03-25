"""Factory for creating geometry encoders."""

from typing import Optional
from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .vggt_encoder import VGGTEncoder
from .pi3_encoder import Pi3Encoder


def create_geometry_encoder(config) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.
    
    Args:
        config: GeometryEncoderConfig instance with encoder configuration.
    Returns:
        Geometry encoder instance
    """

    encoder_type = config.encoder_type.lower()
    if encoder_type == "vggt":
        return VGGTEncoder(config)
    elif encoder_type == "pi3":
        return Pi3Encoder(config)
    else:
        raise ValueError(f"Unknown geometry encoder type: {encoder_type}")


def get_available_encoders():
    """Get list of available encoder types."""
    return ["vggt", "pi3"]
