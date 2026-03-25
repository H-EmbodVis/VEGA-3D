"""PI3 geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class Pi3Encoder(BaseGeometryEncoder):
    """PI3 geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        print("Initializing PI3 Encoder...")
        
        # Lazy import to avoid circular dependencies
        from ..pi3.models.pi3 import Pi3

        # Initialize PI3 model
        self.pi3 = Pi3()

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.pi3.parameters():
                param.requires_grad = False

        self.patch_size = 14
        
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using PI3."""
        
        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # PI3 expects input shape (B, N, C, H, W)
                # where B=batch_size, N=num_frames, C=channels, H=height, W=width
                if images.dim() == 4:  # (N, C, H, W)
                    images = images.unsqueeze(0)  # (1, N, C, H, W)
                
                # Get encoded features from PI3
                # We'll use the decoder output before the heads
                imgs = (images - self.pi3.image_mean) / self.pi3.image_std
                B, N, _, H, W = imgs.shape
                
                # Encode by dinov2
                imgs = imgs.reshape(B*N, _, H, W)
                hidden = self.pi3.encoder(imgs, is_training=True)
                
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
                
                # Use the decoder to get rich features
                features, pos = self.pi3.decode(hidden, N, H, W)

                # Extract features after register tokens
                features = features[:, self.pi3.patch_start_idx:, :]  # Remove register tokens
                
                if B == 1:
                    features = features.squeeze(0)  # Remove batch dimension if added
        
        return features
    

    def get_feature_dim(self) -> int:
        """Get PI3 feature dimension."""
        return self.pi3.dec_embed_dim * 2  # PI3 concatenates two decoder outputs
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)
    

    def load_model(self, model_path: str) -> None:
        """Load pretrained PI3 model."""
        from ..pi3.models.pi3 import Pi3

        self.pi3 = Pi3.from_pretrained(model_path)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.pi3.parameters():
                param.requires_grad = False
