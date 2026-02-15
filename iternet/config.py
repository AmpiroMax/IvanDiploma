from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Paths to raw training data."""

    ie2d_res_path: Path
    ie2_model_path: Path

    # What is stored in the input file as the last column:
    # - "auto": use header (0=app.resistivity, 1=resistance)
    # - "voltage": ΔU (then ρa = ΔU / (L * I))
    # - "resistance": R = ΔU / I (then ρa = K * R)
    # - "rho_a": already apparent resistivity ρa
    value_kind: str = "auto"

    # Injection current in Amperes used to convert ΔU -> ρa.
    # If your file is already normalized by current, set value_kind="resistance" or keep I=1.
    current_a: float = 1.0


@dataclass(frozen=True)
class GridConfig:
    """Grid definition for rasterized targets and model queries."""

    # If provided, these override extents found in the .ie2 file.
    x_min: float | None = None
    x_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None

    # Output grid resolution.
    look_nx: int = 256
    look_nz: int = 128


@dataclass(frozen=True)
class ModelConfig:
    """Perceiver-style set-to-grid segmentation model config."""

    token_dim: int = 64
    latent_dim: int = 128
    num_latents: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    num_classes: int = 10  # will be overwritten from data by default


@dataclass(frozen=True)
class TrainConfig:
    """Training parameters."""

    batch_size: int = 1
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4

    device: str = "cuda"
    log_dir: Path = Path("iternet/runs")

    # Ignore background label 0 by default (unknown/outside bodies)
    ignore_index: int = 0

    # Boundary weighting: pixels near class boundaries get higher loss
    boundary_weight_factor: float = 3.0
    boundary_weight_radius: int = 4

    # Loss weights: ce_weight*CE + dice_weight*Dice + boundary_weight*BoundaryLoss (all terms non-negative)
    ce_weight: float = 3.0
    dice_weight: float = 0.3
    boundary_loss_weight: float = 0.3

    # Logging cadence
    log_every_steps: int = 10
