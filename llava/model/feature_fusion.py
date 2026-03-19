"""Feature fusion modules for combining semantic and generative visual features."""

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""

    # Supported methods: add, concat, weighted, cross_attention,
    # gated, token_gated, instruction_token_gated(context_gated alias), only.
    fusion_method: str = "add"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with residual MLP."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def get_2d_sincos_pos_embed(self, height: int, width: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        return self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
        assert embed_dim % 2 == 0
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000**omega)

        pos = pos.flatten()
        out = torch.einsum("m,d->md", pos, omega)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)

    def forward(self, features_2d: torch.Tensor, features_gen: torch.Tensor, h_grid: int, w_grid: int) -> torch.Tensor:
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_gen)
        value = self.norm1_value(features_gen)

        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True

        pos_embed = self.get_2d_sincos_pos_embed(h_grid, w_grid, self.hidden_size, query.device).to(query.dtype)
        query = query + pos_embed.unsqueeze(0)
        key = key + pos_embed.unsqueeze(0)

        attn_output, _ = self.cross_attention(query, key, value)
        if squeeze_output:
            attn_output = attn_output.squeeze(0)

        x = features_2d + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module with multiple fusion strategies."""

    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        self._build_fusion_layers()

    def _build_fusion_layers(self):
        method = self.config.fusion_method

        if method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)

        elif method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(self.hidden_size, self.config.num_heads, self.config.dropout)
                    for _ in range(self.config.num_layers)
                ]
            )

        elif method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())

        elif method == "token_gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(nn.Linear(self.hidden_size * 2, 1), nn.Sigmoid())

        elif method in {"instruction_token_gated", "context_gated"}:
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.norm_ctx = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Linear(self.hidden_size * 2, 1)
            self.ctx_projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, 1),
            )
            self.context_scale = nn.Parameter(torch.tensor(1.0))
            self.gate_bias = nn.Parameter(torch.tensor(-0.5))

        elif method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_gen = nn.Parameter(torch.tensor(0.5))

        elif method in {"add", "only"}:
            # No trainable fusion parameters.
            pass

        else:
            raise ValueError(
                f"Unknown fusion method: {method}. Supported methods: "
                "add|concat|weighted|cross_attention|gated|token_gated|instruction_token_gated|context_gated|only"
            )

    def _broadcast_instruction_ctx(self, instruction_ctx: torch.Tensor, batch_tokens: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        ctx = instruction_ctx
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        if ctx.dim() != 2 or ctx.size(-1) != self.hidden_size:
            raise ValueError(f"instruction_ctx must be [D] or [B, D], got {tuple(ctx.shape)}")

        ctx = ctx.to(device=device, dtype=dtype)
        if ctx.size(0) == 1 and batch_tokens > 1:
            ctx = ctx.expand(batch_tokens, -1)
        elif ctx.size(0) != batch_tokens:
            ctx = ctx[:1].expand(batch_tokens, -1)
        return ctx

    def forward(
        self,
        features_2d: torch.Tensor,
        features_gen: torch.Tensor,
        instruction_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features_2d: [batch_tokens, seq_len, hidden_size]
            features_gen: [batch_tokens, seq_len, hidden_size]
            instruction_ctx: [hidden_size] or [batch_tokens, hidden_size]
        """

        seq_len = features_2d.size(1)
        h_grid = w_grid = int(math.isqrt(seq_len))
        if h_grid * w_grid != seq_len:
            raise ValueError(f"Sequence length {seq_len} is not a perfect square; cannot form h×w grid.")

        if self.fusion_method == "add":
            return features_2d + features_gen

        if self.fusion_method == "concat":
            f2d = self.norm1(features_2d)
            fgen = self.norm2(features_gen)
            return self.projection(torch.cat([f2d, fgen], dim=-1))

        if self.fusion_method == "cross_attention":
            x = features_2d.view(features_2d.size(0), -1, self.hidden_size)
            fgen = features_gen.view(features_gen.size(0), -1, self.hidden_size)
            for block in self.cross_attn_blocks:
                x = block(x, fgen, h_grid, w_grid)
            return x

        if self.fusion_method == "gated":
            f2d = self.norm1(features_2d)
            fgen = self.norm2(features_gen)
            gate = self.gate_projection(torch.cat([f2d, fgen], dim=-1))
            return gate * f2d + (1.0 - gate) * fgen

        if self.fusion_method == "token_gated":
            f2d = self.norm1(features_2d)
            fgen = self.norm2(features_gen)
            gate = self.gate_projection(torch.cat([f2d, fgen], dim=-1))
            return gate * f2d + (1.0 - gate) * fgen

        if self.fusion_method in {"instruction_token_gated", "context_gated"}:
            f2d = self.norm1(features_2d)
            fgen = self.norm2(features_gen)
            logits = self.gate_projection(torch.cat([f2d, fgen], dim=-1)) + self.gate_bias

            if instruction_ctx is not None:
                ctx = self._broadcast_instruction_ctx(
                    instruction_ctx,
                    batch_tokens=f2d.size(0),
                    dtype=f2d.dtype,
                    device=f2d.device,
                )
                ctx = self.norm_ctx(ctx)
                ctx_logits = self.ctx_projection(ctx).unsqueeze(1)
                logits = logits + self.context_scale * ctx_logits

            gate = torch.sigmoid(logits)
            return gate * f2d + (1.0 - gate) * fgen

        if self.fusion_method == "weighted":
            weight_sum = self.weight_2d + self.weight_gen
            w2d = self.weight_2d / weight_sum
            wgen = self.weight_gen / weight_sum
            return w2d * features_2d + wgen * features_gen

        if self.fusion_method == "only":
            return features_gen

        raise ValueError(f"Unknown fusion method: {self.fusion_method}")
