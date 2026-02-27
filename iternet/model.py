from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Perceiver-style set-to-grid regression model.

    - Encodes variable-length measurements into a fixed set of latents
    - Decodes per-grid-cell values via cross-attention from queries to latents
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
        out_channels: int = 1,
        grid_patches_z: int = 5,
        grid_patches_x: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.grid_patches_z = int(grid_patches_z)
        self.grid_patches_x = int(grid_patches_x)

        self.token_proj = nn.Sequential(
            nn.Linear(in_features, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, latent_dim),
        )

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        self.encoder_layers = nn.ModuleList(
            [CrossAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # Fixed learned queries for each PATCH (no per-pixel grid queries)
        num_patches = self.grid_patches_z * self.grid_patches_x
        self.grid_queries = nn.Parameter(torch.randn(num_patches, latent_dim) * 0.02)

        self.decoder = CrossAttentionBlock(latent_dim, num_heads=num_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, out_channels),
        )

    def forward(self, meas_values_01: torch.Tensor, *, grid_shape: tuple[int, int] = (300, 600)) -> torch.Tensor:
        """
        Args:
            meas_values_01: (B, 1578) or (B, 1578, 1) values in [0,1]
            grid_shape: (Z, X) to reshape the output
        Returns:
            prediction: (B, out_channels, Z, X)
        """

        if meas_values_01.ndim == 2:
            meas_tokens = meas_values_01.unsqueeze(-1)  # (B, 1578, 1)
        elif meas_values_01.ndim == 3:
            meas_tokens = meas_values_01
        else:
            raise ValueError(f"Expected (B,1578) or (B,1578,1), got {tuple(meas_values_01.shape)}")

        b, n_meas, _ = meas_tokens.shape
        if n_meas != 1578:
            raise ValueError(f"Expected 1578 measurements, got {n_meas}")

        tokens = self.token_proj(meas_tokens)  # (B, N_meas, D)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)  # (B, L, D)

        for layer in self.encoder_layers:
            latents = layer(latents, tokens)

        z, x = grid_shape
        if z % self.grid_patches_z != 0 or x % self.grid_patches_x != 0:
            raise ValueError(
                f"grid_shape {grid_shape} must be divisible by patch grid "
                f"({self.grid_patches_z}, {self.grid_patches_x})"
            )
        g = self.grid_patches_z * self.grid_patches_x
        if self.grid_queries.shape[0] != g:
            raise ValueError(
                f"Model grid_queries has {self.grid_queries.shape[0]} patches, but expected {g} "
                f"from ({self.grid_patches_z}*{self.grid_patches_x})."
            )
        queries = self.grid_queries.unsqueeze(0).expand(b, -1, -1)  # (B, G, D)
        decoded = self.decoder(queries, latents)  # (B, G, D)
        pred_flat = self.head(decoded)  # (B, G, out_channels)

        # Patch-grid prediction: (B, C, Pz, Px) -> upsample to (B, C, Z, X)
        pred_patches = pred_flat.transpose(1, 2).reshape(b, self.out_channels, self.grid_patches_z, self.grid_patches_x)
        pred = F.interpolate(pred_patches, size=(z, x), mode="bilinear", align_corners=False)
        return pred

