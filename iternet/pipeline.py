from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iternet.config import DataConfig, GridConfig, ModelConfig, TrainConfig
from iternet.dataset import IternetDataset, SamplePaths, collate_single
from iternet.io import parse_ie2d_res
from iternet.model import IternetUNet
from iternet.preprocessing import PreprocessResult, preprocess_pair
from iternet.train import TrainHistory, train_segmentation
from iternet.viz import (
    Figures,
    plot_measurements_tokens,
    plot_prediction,
    plot_pseudosection,
    plot_two_channel_image,
)


@dataclass(frozen=True)
class RawData:
    ie2d: object
    ie2d_path: Path
    target_matrix_path: Path
    value_kind: str
    current_a: float


@dataclass(frozen=True)
class PreparedData:
    sample: PreprocessResult
    dataset: IternetDataset


def open_training_data(cfg: DataConfig) -> RawData:
    """Step 1: load raw training data from files."""
    ie2d = None
    if Path(cfg.ie2d_res_path).suffix.lower() == ".dat":
        ie2d = parse_ie2d_res(cfg.ie2d_res_path)
    return RawData(
        ie2d=ie2d,
        ie2d_path=Path(cfg.ie2d_res_path),
        target_matrix_path=Path(cfg.target_matrix_path),
        value_kind=cfg.value_kind,
        current_a=cfg.current_a,
    )


def preprocess_data(raw: RawData, grid_cfg: GridConfig) -> PreparedData:
    """Step 2: preprocess raw data into tensors and a dataset."""
    grid_overrides = {
        "x_min": grid_cfg.x_min,
        "x_max": grid_cfg.x_max,
        "z_min": grid_cfg.z_min,
        "z_max": grid_cfg.z_max,
    }
    if raw.ie2d_path.suffix.lower() == ".npz":
        sample = preprocess_pair(
            input_matrix_path=raw.ie2d_path,
            target_matrix_path=raw.target_matrix_path,
            nx=grid_cfg.look_nx,
            nz=grid_cfg.look_nz,
            grid_overrides=grid_overrides,
        )
    else:
        sample = preprocess_pair(
            ie2d=raw.ie2d,
            target_matrix_path=raw.target_matrix_path,
            nx=grid_cfg.look_nx,
            nz=grid_cfg.look_nz,
            grid_overrides=grid_overrides,
            value_kind=raw.value_kind,
            current_a=raw.current_a,
        )

    ds = IternetDataset(
        samples=[
            SamplePaths(
                ie2d_res=raw.ie2d_path,
                target_matrix=raw.target_matrix_path,
            )
        ],
        nx=grid_cfg.look_nx,
        nz=grid_cfg.look_nz,
        grid_overrides={
            "x_min": grid_cfg.x_min,
            "x_max": grid_cfg.x_max,
            "z_min": grid_cfg.z_min,
            "z_max": grid_cfg.z_max,
        },
    )
    # Cache the only sample to avoid re-parsing in notebook.
    ds._cache[0] = sample  # type: ignore[attr-defined]

    return PreparedData(sample=sample, dataset=ds)


def analyze_sample(prep: PreparedData, raw: RawData | None = None) -> Figures:
    """Step 3: visualize inputs/targets for sanity checks."""
    sample = prep.sample
    meas_fig = None
    if raw is not None:
        # User-facing: pseudo cross-section like field software
        if raw.ie2d is not None:
            meas_fig = plot_pseudosection(
                raw.ie2d,
                title="Pseudo cross-section (log10 ρa)",
                value_kind=raw.value_kind,
                current_a=raw.current_a,
            )
        else:
            meas_fig = plot_two_channel_image(sample.input_tensor_01.numpy(), title="Input (matrix_data) — 2 channels")
    else:
        if sample.input_kind == "matrix_data":
            meas_fig = plot_two_channel_image(sample.input_tensor_01.numpy(), title="Input (matrix_data) — 2 channels")
        else:
            # Debug view: plot 1D input vector as an image row
            meas_fig = plot_measurements_tokens(
                sample.input_tensor_01.unsqueeze(1).numpy(), title="Measurement values (1578) [0..1]"
            )

    figs = Figures(
        mask_fig=plot_prediction(
            sample.target_matrix_raw,
            title="Target matrix (raw values)",
            x_coords=sample.x_coords,
            z_coords=sample.z_coords,
        ),
        meas_fig=meas_fig,
        rho_fig=plot_prediction(
            sample.target_matrix_norm.numpy(),
            title="Target matrix (train tensor, compressed domain)",
            x_coords=sample.x_coords,
            z_coords=sample.z_coords,
        ),
    )
    return figs


def init_model(
    prep: PreparedData,
    model_cfg: ModelConfig,
    checkpoint_path: str | Path | None = None,
    *,
    strict: bool = True,
) -> IternetUNet:
    """Step 4: initialize model and optionally load checkpoint weights."""
    # UNet expects 2-channel image input in processed mode
    inp = prep.sample.input_tensor_01
    if inp.ndim != 3 or int(inp.shape[0]) != 2:
        raise ValueError(f"Expected input tensor (2,H,W) for UNet, got {tuple(inp.shape)}")

    model = IternetUNet(
        in_channels=model_cfg.in_channels,
        patch_size=model_cfg.patch_size,
        base_channels=model_cfg.base_channels,
        depth=model_cfg.depth,
        blocks_per_stage=model_cfg.blocks_per_stage,
        stem_blocks=model_cfg.stem_blocks,
        num_heads=model_cfg.num_heads,
        window_size=model_cfg.window_size,
        output_upsample_stages=model_cfg.output_upsample_stages,
        out_channels=model_cfg.out_channels,
    )
    if checkpoint_path is not None:
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict: dict[str, torch.Tensor]
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected state_dict or dict with 'model'/'state_dict' keys."
            )
        model.load_state_dict(state_dict, strict=strict)
    return model


def predict_mask(model: IternetUNet, prep: PreparedData, device: str = "cpu") -> np.ndarray:
    """
    Predict matrix from a model.

    Note:
    - Works the same for trained and untrained models (only quality differs).
    - Returns denormalized float matrix with shape (Z, X).
    """
    model = model.to(device)
    model.eval()
    sample = prep.sample
    with torch.no_grad():
        pred = model(sample.input_tensor_01.unsqueeze(0).to(device), grid_shape=sample.target_matrix_norm.shape)
        pred0 = pred[0, 0].cpu().numpy().astype(np.float32)
    return pred0


def predict_and_visualize(
    model: IternetUNet,
    prep: PreparedData,
    *,
    raw: RawData | None = None,
    device: str = "cpu",
    title: str = "Prediction (matrix)",
) -> tuple[np.ndarray, Figures]:
    """
    Predict and build figures for quick visual inspection.

    If `raw` is provided, the measurement plot will be the pseudosection.
    Otherwise, it will show token debug view.
    """
    pred = predict_mask(model=model, prep=prep, device=device)

    # Keep the same "Step 3" visuals + add prediction
    figs = analyze_sample(prep=prep, raw=raw) if raw is not None else Figures(
        mask_fig=plot_prediction(
            prep.sample.target_matrix_raw,
            title="Target matrix (raw values)",
            x_coords=prep.sample.x_coords,
            z_coords=prep.sample.z_coords,
        ),
        meas_fig=plot_two_channel_image(prep.sample.input_tensor_01.numpy(), title="Input (matrix_data) — 2 channels")
        if prep.sample.input_kind == "matrix_data"
        else plot_measurements_tokens(prep.sample.input_tensor_01.unsqueeze(1).numpy(), title="Measurement values (1578) [0..1]"),
        rho_fig=plot_prediction(
            prep.sample.target_matrix_norm.numpy(),
            title="Target matrix (train tensor, compressed domain)",
            x_coords=prep.sample.x_coords,
            z_coords=prep.sample.z_coords,
        ),
    )
    figs = Figures(
        mask_fig=figs.mask_fig,
        meas_fig=figs.meas_fig,
        rho_fig=figs.rho_fig,
        pred_fig=plot_prediction(
            pred,
            title=title,
            x_coords=prep.sample.x_coords,
            z_coords=prep.sample.z_coords,
        ),
    )
    return pred, figs


def predict_untrained(model: IternetUNet, prep: PreparedData, device: str = "cpu") -> tuple[np.ndarray, Figures]:
    """
    Backward-compatible alias for older notebook cells.

    The prediction logic does not depend on training state.
    """
    return predict_and_visualize(model=model, prep=prep, device=device, title="Prediction (matrix)")


def train_model(model: IternetUNet, prep: PreparedData, train_cfg: TrainConfig) -> TrainHistory:
    """Step 6: run training with TensorBoard logging."""
    loader = DataLoader(
        prep.dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_single,
    )
    config_dict = {f"train_{k}": v for k, v in train_cfg.__dict__.items()}
    history = train_segmentation(
        model=model,
        loader=loader,
        epochs=train_cfg.epochs,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        device=train_cfg.device,
        log_dir=train_cfg.log_dir,
        log_every_steps=train_cfg.log_every_steps,
        mse_weight=train_cfg.mse_weight,
        mae_weight=train_cfg.mae_weight,
        boundary_weight_factor=train_cfg.boundary_weight_factor,
        boundary_weight_radius=train_cfg.boundary_weight_radius,
        boundary_loss_weight=train_cfg.boundary_loss_weight,
        scheduler_name=train_cfg.scheduler_name,
        warmup_epochs=train_cfg.warmup_epochs,
        min_lr_ratio=train_cfg.min_lr_ratio,
        config_dict=config_dict,
    )
    return history

