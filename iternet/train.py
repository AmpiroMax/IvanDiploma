from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from iternet.preprocessing import denormalize_target_matrix
from iternet.viz import plot_target_vs_prediction


@dataclass
class TrainHistory:
    losses: list[float]
    mae: list[float]
    rmse: list[float]
    val_losses: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)


def _compute_regression_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
    """Loss/metrics in normalized target domain."""
    mse = F.mse_loss(pred_norm, target_norm)
    mae = F.l1_loss(pred_norm, target_norm)
    rmse = torch.sqrt(mse + 1e-12)
    # MSE is primary term, MAE stabilizes rare spikes.
    total = mse + 0.1 * mae
    return total, float(mse.item()), float(mae.item()), float(rmse.item())


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, float]:
    """Run validation, return (mean_loss, mean_mae, mean_rmse)."""
    model.eval()
    losses_v: list[float] = []
    maes_v: list[float] = []
    rmses_v: list[float] = []

    with torch.no_grad():
        for meas_values_01, target_matrix_norm, _meta in loader:
            meas_values_01 = meas_values_01.to(device)
            target_matrix_norm = target_matrix_norm.to(device)
            b, z, x = target_matrix_norm.shape
            pred = model(meas_values_01, grid_shape=(z, x))[:, 0]

            loss, _mse, mae, rmse = _compute_regression_loss(pred, target_matrix_norm)
            losses_v.append(float(loss.item()))
            maes_v.append(mae)
            rmses_v.append(rmse)

    model.train()
    return (
        float(np.mean(losses_v)) if losses_v else 0.0,
        float(np.mean(maes_v)) if maes_v else 0.0,
        float(np.mean(rmses_v)) if rmses_v else 0.0,
    )


def train_segmentation(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    log_dir: Path,
    ignore_index: int = 0,  # kept for API compatibility
    log_every_steps: int = 10,
    val_loader: DataLoader | None = None,
    boundary_weight_factor: float = 3.0,  # kept for API compatibility
    boundary_weight_radius: int = 10,  # kept for API compatibility
    ce_weight: float = 1.0,  # kept for API compatibility
    dice_weight: float = 0.3,  # kept for API compatibility
    boundary_loss_weight: float = 0.3,  # kept for API compatibility
    config_dict: dict | None = None,
) -> TrainHistory:
    """Train regression model with optional validation and image logging."""
    _ = (ignore_index, boundary_weight_factor, boundary_weight_radius, ce_weight, dice_weight, boundary_loss_weight)

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
        batch_pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", unit="batch", leave=False)
        for _batch_idx, (meas_values_01, target_matrix_norm, _meta) in batch_pbar:
            meas_values_01 = meas_values_01.to(device)
            target_matrix_norm = target_matrix_norm.to(device)

            b, z, x = target_matrix_norm.shape
            pred = model(meas_values_01, grid_shape=(z, x))[:, 0]
            loss, _mse, mae, rmse = _compute_regression_loss(pred, target_matrix_norm)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            maes.append(mae)
            rmses.append(rmse)
            epoch_losses.append(float(loss.item()))

            batch_pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae:.3f}", rmse=f"{rmse:.3f}")
            if writer is not None and (global_step % log_every_steps == 0):
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/mae", mae, global_step)
                writer.add_scalar("train/rmse", rmse, global_step)
            global_step += 1

        if writer is not None and epoch_losses:
            writer.add_scalar("train/epoch_loss_mean", float(np.mean(epoch_losses)), epoch)

        if val_loader is not None:
            v_loss, v_mae, v_rmse = _run_validation(model=model, loader=val_loader, device=device)
            val_losses.append(v_loss)
            val_mae.append(v_mae)
            val_rmse.append(v_rmse)
            if writer is not None:
                writer.add_scalar("val/loss", v_loss, epoch)
                writer.add_scalar("val/mae", v_mae, epoch)
                writer.add_scalar("val/rmse", v_rmse, epoch)

            model.eval()
            epoch_img_dir = val_images_base / f"epoch_{epoch:04d}"
            epoch_img_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt

            with torch.no_grad():
                global_idx = 0
                for _batch_idx, (meas_values_01, target_matrix_norm, meta) in enumerate(val_loader):
                    meas_values_01 = meas_values_01.to(device)
                    b, z, x = target_matrix_norm.shape
                    pred_b = model(meas_values_01, grid_shape=(z, x))[:, 0]
                    for i in range(b):
                        pred_i_norm = pred_b[i].cpu().numpy()
                        tgt_i_norm = target_matrix_norm[i].cpu().numpy()
                        target_stats = meta.get("target_stats")
                        pred_i = denormalize_target_matrix(pred_i_norm, target_stats) if target_stats else pred_i_norm
                        tgt_i = meta.get("target_matrix_raw", tgt_i_norm)
                        sample_id = meta.get("sample_id", f"sample_{global_idx:04d}")
                        fig = plot_target_vs_prediction(
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
                            img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
                            img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                            img_t = torch.from_numpy(img_arr).float() / 255.0
                            writer.add_image(f"val/epoch_{epoch}/{sample_id}", img_t, epoch, dataformats="HWC")

                        plt.close(fig)
                        global_idx += 1
            model.train()

        postfix = {"loss": f"{np.mean(epoch_losses):.4f}"} if epoch_losses else {"loss": "n/a"}
        if val_rmse:
            postfix["val_rmse"] = f"{val_rmse[-1]:.3f}"
        epoch_pbar.set_postfix(**postfix)

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
