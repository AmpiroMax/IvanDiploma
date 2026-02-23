from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iternet.config import DataConfig, GridConfig, ModelConfig, TrainConfig
from iternet.dataset import IternetDataset, SamplePaths, collate_single
from iternet.io import parse_ie2_model, parse_ie2d_res
from iternet.model import IternetPerceiver
from iternet.preprocessing import PreprocessResult, preprocess_pair
from iternet.train import TrainHistory, train_segmentation
from iternet.viz import (
    Figures,
    plot_mask,
    plot_measurements_tokens,
    plot_prediction,
    plot_pseudosection,
    plot_true_resistivity,
)
from iternet.export_ie2 import ExportConfig, export_prediction_to_ie2


@dataclass(frozen=True)
class RawData:
    ie2d: object
    ie2: object
    ie2d_path: Path
    ie2_path: Path
    value_kind: str
    current_a: float


@dataclass(frozen=True)
class PreparedData:
    sample: PreprocessResult
    dataset: IternetDataset


def open_training_data(cfg: DataConfig) -> RawData:
    """Step 1: load raw training data from files."""
    ie2d = parse_ie2d_res(cfg.ie2d_res_path)
    ie2 = parse_ie2_model(cfg.ie2_model_path)
    return RawData(
        ie2d=ie2d,
        ie2=ie2,
        ie2d_path=Path(cfg.ie2d_res_path),
        ie2_path=Path(cfg.ie2_model_path),
        value_kind=cfg.value_kind,
        current_a=cfg.current_a,
    )


def preprocess_data(raw: RawData, grid_cfg: GridConfig) -> PreparedData:
    """Step 2: preprocess raw data into tensors and a dataset."""
    sample = preprocess_pair(
        ie2d=raw.ie2d,
        ie2=raw.ie2,
        nx=grid_cfg.look_nx,
        nz=grid_cfg.look_nz,
        grid_overrides={
            "x_min": grid_cfg.x_min,
            "x_max": grid_cfg.x_max,
            "z_min": grid_cfg.z_min,
            "z_max": grid_cfg.z_max,
        },
        value_kind=raw.value_kind,
        current_a=raw.current_a,
    )

    ds = IternetDataset(
        samples=[
            SamplePaths(
                ie2d_res=raw.ie2d_path,
                ie2_model=raw.ie2_path,
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
        meas_fig = plot_pseudosection(raw.ie2d, title="Pseudo cross-section (log10 ρa)", value_kind=raw.value_kind, current_a=raw.current_a)
    else:
        # Debug view
        meas_fig = plot_measurements_tokens(sample.meas_tokens.numpy(), title="Measurement tokens (debug)")

    figs = Figures(
        mask_fig=plot_mask(
            sample.target_mask.numpy(),
            title="Target mask (from .ie2 bodies)",
            num_classes=sample.num_classes,
            x_coords=sample.x_coords,
            z_coords=sample.z_coords,
        ),
        meas_fig=meas_fig,
        rho_fig=plot_true_resistivity(
            mask=sample.target_mask.numpy(),
            class_rho=sample.class_rho,
            x_coords=sample.x_coords,
            z_coords=sample.z_coords,
            title="True resistivity ρ(x,z) + mask overlay (from IE2)",
        ),
    )
    return figs


def init_model(
    prep: PreparedData,
    model_cfg: ModelConfig,
    checkpoint_path: str | Path | None = None,
    *,
    strict: bool = True,
) -> IternetPerceiver:
    """Step 4: initialize model and optionally load checkpoint weights."""
    num_classes = prep.sample.num_classes if model_cfg.num_classes <= 0 else max(model_cfg.num_classes, prep.sample.num_classes)
    model = IternetPerceiver(
        in_features=int(prep.sample.meas_tokens.shape[1]),
        token_dim=model_cfg.token_dim,
        latent_dim=model_cfg.latent_dim,
        num_latents=model_cfg.num_latents,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        num_classes=num_classes,
        dropout=model_cfg.dropout,
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


def predict_mask(model: IternetPerceiver, prep: PreparedData, device: str = "cpu") -> np.ndarray:
    """
    Predict segmentation mask from a model.

    Note:
    - Works the same for trained and untrained models (only quality differs).
    - Returns argmax class ids with shape (Z, X).
    """
    model = model.to(device)
    model.eval()
    sample = prep.sample
    with torch.no_grad():
        logits = model(
            sample.meas_tokens.unsqueeze(0).to(device),
            sample.grid_xy.unsqueeze(0).to(device),
            grid_shape=sample.target_mask.shape,
        )
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
    return pred


def predict_and_visualize(
    model: IternetPerceiver,
    prep: PreparedData,
    *,
    raw: RawData | None = None,
    device: str = "cpu",
    title: str = "Prediction (argmax)",
) -> tuple[np.ndarray, Figures]:
    """
    Predict and build figures for quick visual inspection.

    If `raw` is provided, the measurement plot will be the pseudosection.
    Otherwise, it will show token debug view.
    """
    pred = predict_mask(model=model, prep=prep, device=device)

    # Keep the same "Step 3" visuals + add prediction
    figs = analyze_sample(prep=prep, raw=raw) if raw is not None else Figures(
        mask_fig=plot_mask(
            prep.sample.target_mask.numpy(),
            title="Target mask (from .ie2 bodies)",
            num_classes=prep.sample.num_classes,
            x_coords=prep.sample.x_coords,
            z_coords=prep.sample.z_coords,
        ),
        meas_fig=plot_measurements_tokens(prep.sample.meas_tokens.numpy(), title="Measurement tokens (debug)"),
        rho_fig=plot_true_resistivity(
            mask=prep.sample.target_mask.numpy(),
            class_rho=prep.sample.class_rho,
            x_coords=prep.sample.x_coords,
            z_coords=prep.sample.z_coords,
            title="True resistivity ρ(x,z) + mask overlay (from IE2)",
        ),
    )
    figs = Figures(
        mask_fig=figs.mask_fig,
        meas_fig=figs.meas_fig,
        rho_fig=figs.rho_fig,
        pred_fig=plot_prediction(
            pred,
            title=title,
            num_classes=prep.sample.num_classes,
            x_coords=prep.sample.x_coords,
            z_coords=prep.sample.z_coords,
        ),
    )
    return pred, figs


def predict_and_export_ie2(
    *,
    model: IternetPerceiver,
    prep: PreparedData,
    template_ie2_path: str | Path,
    out_path: str | Path,
    device: str = "cpu",
    min_area_cells: int = 50,
    simplify_step: int = 5,
) -> Path:
    """
    Predict a mask with `model` and export it into an IE2 file.

    Note: model training does not change this API.
    """
    pred = predict_mask(model=model, prep=prep, device=device)
    return export_ie2_from_prediction(
        pred_mask=pred,
        prep=prep,
        template_ie2_path=template_ie2_path,
        out_path=out_path,
        min_area_cells=min_area_cells,
        simplify_step=simplify_step,
    )


def predict_untrained(model: IternetPerceiver, prep: PreparedData, device: str = "cpu") -> tuple[np.ndarray, Figures]:
    """
    Backward-compatible alias for older notebook cells.

    The prediction logic does not depend on training state.
    """
    return predict_and_visualize(model=model, prep=prep, device=device, title="Prediction (argmax)")


def train_model(model: IternetPerceiver, prep: PreparedData, train_cfg: TrainConfig) -> TrainHistory:
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
        ignore_index=train_cfg.ignore_index,
        log_every_steps=train_cfg.log_every_steps,
        boundary_weight_factor=train_cfg.boundary_weight_factor,
        boundary_weight_radius=train_cfg.boundary_weight_radius,
        ce_weight=train_cfg.ce_weight,
        dice_weight=train_cfg.dice_weight,
        boundary_loss_weight=train_cfg.boundary_loss_weight,
        config_dict=config_dict,
    )
    return history


def export_ie2_from_prediction(
    *,
    pred_mask: np.ndarray,
    prep: PreparedData,
    template_ie2_path: str | Path,
    out_path: str | Path,
    min_area_cells: int = 50,
    simplify_step: int = 5,
) -> Path:
    """Export predicted segmentation mask into an IE2 file."""
    cfg = ExportConfig(min_area_cells=min_area_cells, simplify_step=simplify_step)
    sample = prep.sample
    return export_prediction_to_ie2(
        pred_mask=pred_mask,
        x_coords=sample.x_coords,
        z_coords=sample.z_coords,
        class_rho=sample.class_rho,
        template_ie2_path=template_ie2_path,
        out_path=out_path,
        config=cfg,
    )

