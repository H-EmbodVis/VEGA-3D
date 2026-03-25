"""Base classes for geometry encoders."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GeometryEncoderConfig:
    """Configuration for geometry encoders."""
    encoder_type: str = "vggt"  # "vggt", "pi3", etc.
    model_path: Optional[str] = None
    reference_frame: str = "first"  # "first" or "last"
    feature_dim: int = 2048  # Will be overridden by encoder's get_feature_dim()
    freeze_encoder: bool = True
    
    # Encoder-specific configs
    encoder_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.encoder_kwargs is None:
            self.encoder_kwargs = {}


class BaseGeometryEncoder(ABC, nn.Module):
    """Base class for geometry encoders like VGGT, Pi3, etc."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__()
        self.config = config
        self.reference_frame = config.reference_frame
        self.freeze_encoder = config.freeze_encoder
        
    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to geometry features.
        
        Args:
            images: Input images tensor
            
        Returns:
            Geometry features tensor
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        pass
