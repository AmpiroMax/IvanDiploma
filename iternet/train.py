from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from iternet.viz import plot_abcd_components, plot_target_vs_prediction


@dataclass
class TrainHistory:
    losses: list[float]
    mae: list[float]
    rmse: list[float]
    val_losses: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)


def _boundary_mask_from_target(
    target: torch.Tensor,
    *,
    radius: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute a boundary mask from a continuous target by detecting high gradients.

    Args:
        target: (B, Z, X)
        radius: dilation radius in pixels (>=0)
    Returns:
        mask: (B, Z, X) float in {0,1}
    """
    if target.ndim != 3:
        raise ValueError(f"Expected target (B,Z,X), got {tuple(target.shape)}")

    # simple finite-difference gradients
    dx = target[:, :, 1:] - target[:, :, :-1]  # (B,Z,X-1)
    dz = target[:, 1:, :] - target[:, :-1, :]  # (B,Z-1,X)
    dx = F.pad(dx, (0, 1, 0, 0))  # pad X to (B,Z,X)
    dz = F.pad(dz, (0, 0, 0, 1))  # pad Z to (B,Z,X)

    grad = torch.sqrt(dx * dx + dz * dz + eps)  # (B,Z,X)

    # per-sample threshold: mean + std (robust enough, no quantiles needed)
    mean = grad.flatten(1).mean(dim=1).view(-1, 1, 1)
    std = grad.flatten(1).std(dim=1).view(-1, 1, 1)
    thr = mean + std
    edge = (grad > thr).to(dtype=target.dtype)  # (B,Z,X)

    r = int(max(radius, 0))
    if r == 0:
        return edge

    # dilate edges to include neighborhood
    edge4 = edge.unsqueeze(1)  # (B,1,Z,X)
    dil = F.max_pool2d(edge4, kernel_size=2 * r + 1, stride=1, padding=r)
    return dil[:, 0]


def _compute_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    w_mse: float = 1.0,
    w_mae: float = 0.05,
    boundary_weight_factor: float = 3.0,
    boundary_weight_radius: int = 4,
    boundary_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss/metrics in raw target domain (no normalization), with component logging support."""
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    rmse = torch.sqrt(mse + 1e-12)

    # Boundary MAE: compute mask from target gradients, then MAE on boundary region
    if boundary_loss_weight != 0.0:
        mask = _boundary_mask_from_target(target, radius=boundary_weight_radius)  # (B,Z,X)
        err = torch.abs(pred - target)
        num = (err * mask).sum()
        den = mask.sum().clamp_min(1.0)
        boundary_mae = num / den
    else:
        boundary_mae = torch.zeros((), device=pred.device, dtype=pred.dtype)

    mse_w = float(w_mse) * mse
    mae_w = float(w_mae) * mae
    boundary_scaled = float(boundary_weight_factor) * boundary_mae
    boundary_w = float(boundary_loss_weight) * boundary_scaled
    total = mse_w + mae_w + boundary_w

    metrics = {
        # raw components
        "mse_raw": float(mse.item()),
        "mae_raw": float(mae.item()),
        "rmse": float(rmse.item()),
        "boundary_mae_raw": float(boundary_mae.item()),
        # weighted components (after multiplying by coefficients)
        "mse_w": float(mse_w.item()),
        "mae_w": float(mae_w.item()),
        "boundary_mae_scaled": float(boundary_scaled.item()),
        "boundary_w": float(boundary_w.item()),
        # sum
        "total": float(total.item()),
        # coefficients (for dashboard clarity)
        "w_mse": float(w_mse),
        "w_mae": float(w_mae),
        "boundary_weight_factor": float(boundary_weight_factor),
        "boundary_loss_weight": float(boundary_loss_weight),
        "boundary_weight_radius": float(boundary_weight_radius),
    }
    return total, metrics


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    mse_weight: float,
    mae_weight: float,
    boundary_weight_factor: float,
    boundary_weight_radius: int,
    boundary_loss_weight: float,
) -> dict[str, float]:
    """Run validation, return mean metrics dict (same keys as training component metrics)."""
    model.eval()
    agg: dict[str, list[float]] = {}

    with torch.no_grad():
        for inputs_01, target_matrix_norm, _meta in loader:
            inputs_01 = inputs_01.to(device)
            target_matrix_norm = target_matrix_norm.to(device)
            b, z, x = target_matrix_norm.shape
            pred = model(inputs_01, grid_shape=(z, x))[:, 0]

            loss, m = _compute_regression_loss(
                pred,
                target_matrix_norm,
                w_mse=mse_weight,
                w_mae=mae_weight,
                boundary_weight_factor=boundary_weight_factor,
                boundary_weight_radius=boundary_weight_radius,
                boundary_loss_weight=boundary_loss_weight,
            )
            _ = loss
            for k, v in m.items():
                agg.setdefault(k, []).append(float(v))

    model.train()
    return {k: (float(np.mean(vs)) if vs else 0.0) for k, vs in agg.items()}


def train_segmentation(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    log_dir: Path,
    log_every_steps: int = 10,
    val_loader: DataLoader | None = None,
    boundary_weight_factor: float = 3.0,
    boundary_weight_radius: int = 10,
    boundary_loss_weight: float = 0.3,
    mse_weight: float = 1.0,
    mae_weight: float = 0.05,
    config_dict: dict | None = None,
) -> TrainHistory:
    """Train regression model with optional validation and image logging."""

    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    actual_log_dir = Path(log_dir) / run_name
    writer = None
    if SummaryWriter is not None:
        actual_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(actual_log_dir))
        print(f"TensorBoard logs: {actual_log_dir}")
        if config_dict:
            config_text = "\n".join(f"{k}: {v}" for k, v in sorted(config_dict.items()))
            writer.add_text("config", config_text, 0)
            print("Config logged to TensorBoard")

    losses: list[float] = []
    maes: list[float] = []
    rmses: list[float] = []
    val_losses: list[float] = []
    val_mae: list[float] = []
    val_rmse: list[float] = []
    val_images_base = actual_log_dir / "val_images"

    global_step = 0
    epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="ep")
    for epoch in epoch_pbar:
        epoch_losses: list[float] = []
        epoch_mses_raw: list[float] = []
        epoch_maes_raw: list[float] = []
        epoch_rmses: list[float] = []
        epoch_mses_w: list[float] = []
        epoch_maes_w: list[float] = []
        epoch_boundary_raw: list[float] = []
        epoch_boundary_scaled: list[float] = []
        epoch_boundary_w: list[float] = []
        batch_pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", unit="batch", leave=False)
        for _batch_idx, (inputs_01, target_matrix_norm, _meta) in batch_pbar:
            inputs_01 = inputs_01.to(device)
            target_matrix_norm = target_matrix_norm.to(device)

            b, z, x = target_matrix_norm.shape
            pred = model(inputs_01, grid_shape=(z, x))[:, 0]
            loss, m = _compute_regression_loss(
                pred,
                target_matrix_norm,
                w_mse=mse_weight,
                w_mae=mae_weight,
                boundary_weight_factor=boundary_weight_factor,
                boundary_weight_radius=boundary_weight_radius,
                boundary_loss_weight=boundary_loss_weight,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            maes.append(m["mae_raw"])
            rmses.append(m["rmse"])
            epoch_losses.append(float(loss.item()))
            epoch_mses_raw.append(m["mse_raw"])
            epoch_maes_raw.append(m["mae_raw"])
            epoch_rmses.append(m["rmse"])
            epoch_mses_w.append(m["mse_w"])
            epoch_maes_w.append(m["mae_w"])
            epoch_boundary_raw.append(m["boundary_mae_raw"])
            epoch_boundary_scaled.append(m["boundary_mae_scaled"])
            epoch_boundary_w.append(m["boundary_w"])

            batch_pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{m['mae_raw']:.3f}", rmse=f"{m['rmse']:.3f}")
            if writer is not None and (global_step % log_every_steps == 0):
                # raw components
                writer.add_scalar("train/loss_components/mse_raw", m["mse_raw"], global_step)
                writer.add_scalar("train/loss_components/mae_raw", m["mae_raw"], global_step)
                writer.add_scalar("train/loss_components/rmse", m["rmse"], global_step)
                writer.add_scalar("train/loss_components/boundary_mae_raw", m["boundary_mae_raw"], global_step)
                # weighted components
                writer.add_scalar("train/loss_components/mse_weighted", m["mse_w"], global_step)
                writer.add_scalar("train/loss_components/mae_weighted", m["mae_w"], global_step)
                writer.add_scalar("train/loss_components/boundary_mae_scaled", m["boundary_mae_scaled"], global_step)
                writer.add_scalar("train/loss_components/boundary_weighted", m["boundary_w"], global_step)
                # sum
                writer.add_scalar("train/loss_components/total", m["total"], global_step)
                # keep backward-compatible main scalars
                writer.add_scalar("train/loss", m["total"], global_step)
                writer.add_scalar("train/mae", m["mae_raw"], global_step)
                writer.add_scalar("train/rmse", m["rmse"], global_step)
            global_step += 1

        if writer is not None and epoch_losses:
            # epoch means for component dashboard
            writer.add_scalar("train_epoch/loss_components/mse_raw_mean", float(np.mean(epoch_mses_raw)), epoch)
            writer.add_scalar("train_epoch/loss_components/mae_raw_mean", float(np.mean(epoch_maes_raw)), epoch)
            writer.add_scalar("train_epoch/loss_components/rmse_mean", float(np.mean(epoch_rmses)), epoch)
            writer.add_scalar("train_epoch/loss_components/boundary_mae_raw_mean", float(np.mean(epoch_boundary_raw)), epoch)
            writer.add_scalar("train_epoch/loss_components/mse_weighted_mean", float(np.mean(epoch_mses_w)), epoch)
            writer.add_scalar("train_epoch/loss_components/mae_weighted_mean", float(np.mean(epoch_maes_w)), epoch)
            writer.add_scalar("train_epoch/loss_components/boundary_mae_scaled_mean", float(np.mean(epoch_boundary_scaled)), epoch)
            writer.add_scalar("train_epoch/loss_components/boundary_weighted_mean", float(np.mean(epoch_boundary_w)), epoch)
            writer.add_scalar("train_epoch/loss_components/total_mean", float(np.mean(epoch_losses)), epoch)
            # keep backward-compatible name
            writer.add_scalar("train/epoch_loss_mean", float(np.mean(epoch_losses)), epoch)

        if val_loader is not None:
            vm = _run_validation(
                model=model,
                loader=val_loader,
                device=device,
                mse_weight=mse_weight,
                mae_weight=mae_weight,
                boundary_weight_factor=boundary_weight_factor,
                boundary_weight_radius=boundary_weight_radius,
                boundary_loss_weight=boundary_loss_weight,
            )
            val_losses.append(vm.get("total", 0.0))
            val_mae.append(vm.get("mae_raw", 0.0))
            val_rmse.append(vm.get("rmse", 0.0))
            if writer is not None:
                # Validation: keep simple + add component-style group (raw target domain)
                writer.add_scalar("val/loss_components/mse_raw_mean", vm.get("mse_raw", 0.0), epoch)
                writer.add_scalar("val/loss_components/mae_raw_mean", vm.get("mae_raw", 0.0), epoch)
                writer.add_scalar("val/loss_components/rmse_mean", vm.get("rmse", 0.0), epoch)
                writer.add_scalar("val/loss_components/boundary_mae_raw_mean", vm.get("boundary_mae_raw", 0.0), epoch)
                writer.add_scalar("val/loss_components/mse_weighted_mean", vm.get("mse_w", 0.0), epoch)
                writer.add_scalar("val/loss_components/mae_weighted_mean", vm.get("mae_w", 0.0), epoch)
                writer.add_scalar("val/loss_components/boundary_weighted_mean", vm.get("boundary_w", 0.0), epoch)
                writer.add_scalar("val/loss_components/total_mean", vm.get("total", 0.0), epoch)
                writer.add_scalar("val/loss", vm.get("total", 0.0), epoch)
                writer.add_scalar("val/mae", vm.get("mae_raw", 0.0), epoch)
                writer.add_scalar("val/rmse", vm.get("rmse", 0.0), epoch)

            model.eval()
            epoch_img_dir = val_images_base / f"epoch_{epoch:04d}"
            epoch_img_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt

            with torch.no_grad():
                global_idx = 0
                for _batch_idx, (inputs_01, target_matrix_norm, meta) in enumerate(val_loader):
                    inputs_01 = inputs_01.to(device)
                    b, z, x = target_matrix_norm.shape
                    pred_and_abcd = model(inputs_01, grid_shape=(z, x), return_abcd=True)
                    pred_b = pred_and_abcd[0][:, 0]
                    abcd = pred_and_abcd[1]
                    for i in range(b):
                        pred_i = pred_b[i].detach().cpu().numpy()
                        tgt_i = meta.get("target_matrix_raw", target_matrix_norm[i].cpu().numpy())
                        sample_id = meta.get("sample_id", f"sample_{global_idx:04d}")
                        fig: Any = plot_target_vs_prediction(
                            tgt_i,
                            pred_i,
                            title=f"Epoch {epoch} - {sample_id}",
                            x_coords=meta.get("x_coords"),
                            z_coords=meta.get("z_coords"),
                        )
                        out_path = epoch_img_dir / f"{sample_id}.png"
                        fig.savefig(out_path, dpi=100, bbox_inches="tight")

                        if writer is not None:
                            fig.canvas.draw()
                            # Backend-agnostic RGB extraction (works for TkAgg/Agg).
                            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                            # buffer_rgba returns (H, W, 4) in RGBA; drop alpha
                            img_arr = buf[..., :3].copy()
                            img_t = torch.from_numpy(img_arr).float() / 255.0
                            writer.add_image(f"val/epoch_{epoch}/{sample_id}", img_t, epoch, dataformats="HWC")

                        plt.close(fig)

                        # A/B/C/D coefficient maps visualization
                        a_i = abcd["a"][i, 0].detach().cpu().numpy()
                        b_i = abcd["b"][i, 0].detach().cpu().numpy()
                        c_i = abcd["c"][i, 0].detach().cpu().numpy()
                        d_i = abcd["d"][i, 0].detach().cpu().numpy()
                        fig_abcd: Any = plot_abcd_components(
                            a=a_i,
                            b=b_i,
                            c=c_i,
                            d=d_i,
                            title=f"Epoch {epoch} - {sample_id} (A/B/C/D)",
                            x_coords=meta.get("x_coords"),
                            z_coords=meta.get("z_coords"),
                        )
                        out_path_abcd = epoch_img_dir / f"{sample_id}_abcd.png"
                        fig_abcd.savefig(out_path_abcd, dpi=100, bbox_inches="tight")

                        if writer is not None:
                            fig_abcd.canvas.draw()
                            buf2 = np.asarray(fig_abcd.canvas.buffer_rgba(), dtype=np.uint8)
                            img_arr2 = buf2[..., :3].copy()
                            img_t2 = torch.from_numpy(img_arr2).float() / 255.0
                            writer.add_image(f"val_abcd/epoch_{epoch}/{sample_id}", img_t2, epoch, dataformats="HWC")

                        plt.close(fig_abcd)
                        global_idx += 1
            model.train()

        postfix = {"loss": f"{np.mean(epoch_losses):.4f}"} if epoch_losses else {"loss": "n/a"}
        if val_rmse:
            postfix["val_rmse"] = f"{val_rmse[-1]:.3f}"
        epoch_pbar.set_postfix(postfix)

    if writer is not None:
        writer.flush()
        writer.close()

    return TrainHistory(
        losses=losses,
        mae=maes,
        rmse=rmses,
        val_losses=val_losses,
        val_mae=val_mae,
        val_rmse=val_rmse,
    )
