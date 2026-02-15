from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block:
    query attends to key/value.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(query)
        kvn = self.kv_norm(kv)
        out, _ = self.attn(q, kvn, kvn, need_weights=False)
        x = query + out
        x = x + self.ff(x)
        return x


class IternetPerceiver(nn.Module):
    """
    Perceiver-style set-to-grid segmentation model.

    - Encodes variable-length measurements into a fixed set of latents
    - Decodes per-grid-cell class logits via cross-attention from queries to latents
    """

    def __init__(
        self,
        *,
        in_features: int,
        token_dim: int,
        latent_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.token_proj = nn.Sequential(
            nn.Linear(in_features, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, latent_dim),
        )

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        self.encoder_layers = nn.ModuleList(
            [CrossAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # Query embedding from normalized (x,z) -> latent_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.decoder = CrossAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, meas_tokens: torch.Tensor, grid_xy: torch.Tensor, *, grid_shape: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            meas_tokens: (B, N_meas, F)
            grid_xy: (B, N_grid, 2) normalized [-1,1]
            grid_shape: (Z, X) to reshape the output
        Returns:
            logits: (B, C, Z, X)
        """

        b, n_meas, _ = meas_tokens.shape
        _, n_grid, _ = grid_xy.shape

        tokens = self.token_proj(meas_tokens)  # (B, N_meas, D)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)  # (B, L, D)

        for layer in self.encoder_layers:
            latents = layer(latents, tokens)

        queries = self.query_mlp(grid_xy)  # (B, G, D)
        decoded = self.decoder(queries, latents)  # (B, G, D)
        logits_flat = self.head(decoded)  # (B, G, C)

        z, x = grid_shape
        logits = logits_flat.transpose(1, 2).reshape(b, self.num_classes, z, x)
        return logits

