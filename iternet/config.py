from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Paths to training data.

    Input can be:
    - legacy `.dat` (ERT measurements)
    - processed `.npz` (matrix_data with 2 channels)
    """

    ie2d_res_path: Path
    target_matrix_path: Path

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
    """Grid definition for target matrix and model queries."""

    # Physical extents for X/Z coordinates used by query grid and visualization.
    x_min: float | None = None
    x_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None

    # Output matrix resolution.
    # By current task definition target matrix is 300x600 (Z,X).
    look_nx: int = 600
    look_nz: int = 300


@dataclass(frozen=True)
class ModelConfig:
    """Patch-UNet + Swin-style attention + digit-polynomial projection."""

    # Processed input: matrix_data -> (B, 2, 29, 47)
    in_channels: int = 2

    # Patch embedding (Swin/ViT-like): split input into patches using stride-conv
    # patch_size=2 works well for (29,47) with padding to multiples.
    patch_size: int = 2

    # U-Net width multiplier (32 is a good default; deeper model benefits from 48/64)
    base_channels: int = 32

    # Depth of U-Net in patch space (number of downsamplings)
    depth: int = 4

    # How many residual conv blocks per stage (encoder/decoder/bottleneck)
    blocks_per_stage: int = 3

    # Extra conv blocks right after patch embedding
    stem_blocks: int = 2

    # Swin-style local self-attention parameters.
    num_heads: int = 4
    window_size: int = 4

    # Progressive decoder to final (Z, X) resolution.
    output_upsample_stages: int = 3

    # The polynomial projection outputs a single scalar per pixel:
    # y = A*1000 + B*100 + C*10 + D
    out_channels: int = 1


@dataclass(frozen=True)
class TrainConfig:
    """Training parameters (regression)."""

    batch_size: int = 1
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4

    device: str = "cuda"
    log_dir: Path = Path("iternet/runs")

    # Learning-rate schedule.
    scheduler_name: str = "cosine"
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.1

    # Loss weights (total = mse_weight*MSE + mae_weight*MAE + boundary_loss_weight*BoundaryMAE)
    mse_weight: float = 1.0
    mae_weight: float = 0.05

    # Boundary loss: focus on high-gradient areas in the target (edges), dilated by radius.
    # The boundary term itself is MAE computed on the boundary mask.
    boundary_weight_factor: float = 3.0
    boundary_weight_radius: int = 4
    boundary_loss_weight: float = 0.3

    # Logging cadence
    log_every_steps: int = 10
