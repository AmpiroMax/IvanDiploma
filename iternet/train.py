from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from iternet.viz import plot_target_vs_prediction


@dataclass
class TrainHistory:
    losses: list[float]
    acc: list[float]
    miou: list[float]
    val_losses: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_miou: list[float] = field(default_factory=list)


def _fast_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 0) -> float:
    """
    Mean IoU over classes excluding ignore_index.
    pred/target: (Z,X) int64
    """
    ious: list[float] = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def _class_weights_from_mask(mask: torch.Tensor, num_classes: int, ignore_index: int = 0) -> torch.Tensor:
    """
    Inverse-frequency weights for CrossEntropyLoss.
    mask: (Z,X) int64
    """
    flat = mask.view(-1)
    valid = flat != ignore_index
    flat = flat[valid]
    if flat.numel() == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=mask.device)

    counts = torch.bincount(flat, minlength=num_classes).float()
    counts = counts[:num_classes]
    if counts.numel() < num_classes:
        counts = torch.nn.functional.pad(counts, (0, num_classes - counts.numel()), value=1.0)
    counts[ignore_index] = 0.0
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    inv[ignore_index] = 0.0
    mean = inv[inv > 0].mean()
    w = inv / (mean + 1e-9)
    return w[:num_classes]


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 0) -> torch.Tensor:
    """
    Multi-class Dice loss: 1 - mean(Dice_c) over classes.
    logits: (B, C, Z, X), target: (B, Z, X) int64
    """
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = (target != ignore_index).unsqueeze(1).expand_as(probs)
    probs = probs * valid
    target_onehot = target_onehot * valid

    smooth = 1e-6
    dice_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        p = probs[:, c]
        t = target_onehot[:, c]
        inter = (p * t).sum()
        union = p.sum() + t.sum() + smooth
        dice_per_class.append((2 * inter + smooth) / union)
    if not dice_per_class:
        return torch.tensor(0.0, device=logits.device)
    return 1.0 - torch.stack(dice_per_class).mean()


def _boundary_loss_kervadec(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    Boundary loss (Kervadec et al.): L = sum_p sum_c (p_c - y_c) * phi_c.
    phi_c = signed distance (inside +, outside -).
    Raw sum can be negative; we return max(0, L) / n_pixels for non-negative, scale-invariant loss.
    logits: (B, C, Z, X), target: (B, Z, X) int64
    """
    probs = F.softmax(logits, dim=1)
    device = logits.device
    b = logits.shape[0]
    loss = torch.tensor(0.0, device=device, dtype=logits.dtype)

    for bi in range(b):
        t = target[bi].cpu().numpy()
        valid = (t != ignore_index).astype(np.float32)
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            mask_c = (t == cls).astype(np.float32)
            dist_inside = distance_transform_edt(mask_c > 0.5)
            dist_outside = distance_transform_edt(mask_c < 0.5)
            phi = torch.from_numpy((dist_inside - dist_outside) * valid).to(device).float()
            p_c = probs[bi, cls]
            y_c = torch.from_numpy(mask_c * valid).to(device)
            loss = loss + ((p_c - y_c) * phi).sum()

    raw = loss / max(b, 1)
    # Ensure non-negative: use |L| so total loss stays >= 0
    return torch.abs(raw)


def _boundary_weight_map(mask: torch.Tensor, ignore_index: int = 0, factor: float = 3.0, radius: int = 1) -> torch.Tensor:
    """
    Compute pixel weights: boundary pixels get multiplied by `factor`.
    Uses simple neighbor-difference edge detection + dilation.
    mask: (Z,X) int64
    returns: (Z,X) float32
    """
    t = mask
    valid = t != ignore_index

    # Edge if neighbor label differs (4-neighborhood)
    edge = torch.zeros_like(t, dtype=torch.bool)
    edge |= (t != torch.roll(t, shifts=1, dims=0))
    edge |= (t != torch.roll(t, shifts=-1, dims=0))
    edge |= (t != torch.roll(t, shifts=1, dims=1))
    edge |= (t != torch.roll(t, shifts=-1, dims=1))
    edge &= valid

    # Dilate edges by maxpool to cover near-boundary pixels
    e = edge.float().unsqueeze(0).unsqueeze(0)  # (1,1,Z,X)
    k = 2 * radius + 1
    dil = F.max_pool2d(e, kernel_size=k, stride=1, padding=radius)
    dil = (dil[0, 0] > 0.0)

    w = torch.ones_like(t, dtype=torch.float32)
    w[dil] = factor
    w[~valid] = 0.0
    return w


def _compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    boundary_factor: float,
    boundary_radius: int,
    ce_weight: float,
    dice_weight: float,
    boundary_loss_weight: float,
    device: str,
) -> tuple[torch.Tensor, float, float, float]:
    """Compute combined ce_weight*CE + dice_weight*Dice + boundary_weight*Boundary. All terms non-negative."""
    cw = _class_weights_from_mask(target[0].long(), num_classes=num_classes, ignore_index=ignore_index).to(device)
    ce_fn = torch.nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index, reduction="none")
    per_pix = ce_fn(logits, target.long())
    bw = _boundary_weight_map(
        target[0].long(), ignore_index=ignore_index, factor=boundary_factor, radius=boundary_radius
    ).to(device).unsqueeze(0)
    ce_loss = (per_pix * bw).sum() / (bw.sum() + 1e-9)

    dice_part = torch.tensor(0.0, device=device)
    if dice_weight > 0:
        dice_part = _dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)

    boundary_part = torch.tensor(0.0, device=device)
    if boundary_loss_weight > 0:
        boundary_part = _boundary_loss_kervadec(logits, target, num_classes=num_classes, ignore_index=ignore_index)

    total = ce_weight * ce_loss + dice_weight * dice_part + boundary_loss_weight * boundary_part
    return total, float(ce_loss.item()), float(dice_part.item()), float(boundary_part.item())


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    ignore_index: int,
    boundary_factor: float,
    boundary_radius: int,
    ce_weight: float,
    dice_weight: float,
    boundary_loss_weight: float,
) -> tuple[float, float, float]:
    """Run validation, return (mean_loss, mean_acc, mean_miou)."""
    model.eval()
    losses_v: list[float] = []
    accs_v: list[float] = []
    mious_v: list[float] = []
    num_classes = None

    with torch.no_grad():
        for meas_tokens, grid_xy, target_mask, meta in loader:
            meas_tokens = meas_tokens.to(device)
            grid_xy = grid_xy.to(device)
            target_mask = target_mask.to(device)
            b, z, x = target_mask.shape
            logits = model(meas_tokens, grid_xy, grid_shape=(z, x))
            num_classes = int(logits.shape[1])

            loss, _, _, _ = _compute_loss(
                logits,
                target_mask,
                num_classes=num_classes,
                ignore_index=ignore_index,
                boundary_factor=boundary_factor,
                boundary_radius=boundary_radius,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
                boundary_loss_weight=boundary_loss_weight,
                device=device,
            )

            pred = torch.argmax(logits, dim=1)
            valid = target_mask != ignore_index
            correct = ((pred == target_mask) & valid).sum().item()
            total = valid.sum().item() + 1e-9
            acc = float(correct / total)
            miou = _fast_iou(pred[0].cpu(), target_mask[0].cpu(), num_classes=num_classes, ignore_index=ignore_index)

            losses_v.append(float(loss.item()))
            accs_v.append(acc)
            mious_v.append(miou)

    model.train()
    return (
        float(np.mean(losses_v)) if losses_v else 0.0,
        float(np.mean(accs_v)) if accs_v else 0.0,
        float(np.mean(mious_v)) if mious_v else 0.0,
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
    ignore_index: int = 0,
    log_every_steps: int = 10,
    val_loader: DataLoader | None = None,
    boundary_weight_factor: float = 3.0,
    boundary_weight_radius: int = 10,
    ce_weight: float = 1.0,
    dice_weight: float = 0.3,
    boundary_loss_weight: float = 0.3,
    config_dict: dict | None = None,
) -> TrainHistory:
    """
    Train with optional validation after each epoch.
    If val_loader is set, saves target vs prediction images next to TF logs, in epoch subdirs.
    """
    model = model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # TensorBoard: unique subdir per run (date-time) so runs don't overlap
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    actual_log_dir = Path(log_dir) / run_name
    writer = None
    if SummaryWriter is not None:
        actual_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(actual_log_dir))
        print(f"TensorBoard logs: {actual_log_dir}")

        # Log config at start for experiment comparison and reproducibility
        if config_dict:
            config_text = "\n".join(f"{k}: {v}" for k, v in sorted(config_dict.items()))
            writer.add_text("config", config_text, 0)
            print("Config logged to TensorBoard")

    losses: list[float] = []
    accs: list[float] = []
    mious: list[float] = []
    val_losses: list[float] = []
    val_acc: list[float] = []
    val_miou: list[float] = []

    # Val images: next to TF logs, in epoch subdirs (val_images/epoch_0000/, epoch_0001/, ...)
    val_images_base = actual_log_dir / "val_images"

    global_step = 0
    epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="ep")
    for epoch in epoch_pbar:
        epoch_losses: list[float] = []
        batch_pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", unit="batch", leave=False)
        for batch_idx, (meas_tokens, grid_xy, target_mask, meta) in batch_pbar:
            meas_tokens = meas_tokens.to(device)
            grid_xy = grid_xy.to(device)
            target_mask = target_mask.to(device)

            b, z, x = target_mask.shape
            logits = model(meas_tokens, grid_xy, grid_shape=(z, x))

            num_classes = int(logits.shape[1])
            loss, _, _, _ = _compute_loss(
                logits,
                target_mask,
                num_classes=num_classes,
                ignore_index=ignore_index,
                boundary_factor=boundary_weight_factor,
                boundary_radius=boundary_weight_radius,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
                boundary_loss_weight=boundary_loss_weight,
                device=device,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                valid = target_mask != ignore_index
                correct = ((pred == target_mask) & valid).sum().item()
                total = valid.sum().item() + 1e-9
                acc = float(correct / total)
                miou = _fast_iou(pred[0].cpu(), target_mask[0].cpu(), num_classes=num_classes, ignore_index=ignore_index)

            losses.append(float(loss.item()))
            accs.append(acc)
            mious.append(miou)
            epoch_losses.append(float(loss.item()))

            batch_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", miou=f"{miou:.3f}")

            if writer is not None and (global_step % log_every_steps == 0):
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/acc", acc, global_step)
                writer.add_scalar("train/miou", miou, global_step)

            global_step += 1

        if writer is not None:
            writer.add_scalar("train/epoch_loss_mean", float(np.mean(epoch_losses)), epoch)

        # Validation after each epoch
        if val_loader is not None:
            v_loss, v_acc, v_miou = _run_validation(
                model=model,
                loader=val_loader,
                device=device,
                ignore_index=ignore_index,
                boundary_factor=boundary_weight_factor,
                boundary_radius=boundary_weight_radius,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
                boundary_loss_weight=boundary_loss_weight,
            )
            val_losses.append(v_loss)
            val_acc.append(v_acc)
            val_miou.append(v_miou)
            if writer is not None:
                writer.add_scalar("val/loss", v_loss, epoch)
                writer.add_scalar("val/acc", v_acc, epoch)
                writer.add_scalar("val/miou", v_miou, epoch)

            # Save test images: next to TF logs, in epoch subdirs; add to TensorBoard
            if val_loader is not None:
                model.eval()
                epoch_img_dir = val_images_base / f"epoch_{epoch:04d}"
                epoch_img_dir.mkdir(parents=True, exist_ok=True)
                import matplotlib.pyplot as plt

                with torch.no_grad():
                    global_idx = 0
                    for batch_idx, (meas_tokens, grid_xy, target_mask, meta) in enumerate(val_loader):
                        meas_tokens = meas_tokens.to(device)
                        grid_xy = grid_xy.to(device)
                        b, z, x = target_mask.shape
                        logits = model(meas_tokens, grid_xy, grid_shape=(z, x))
                        pred_b = torch.argmax(logits, dim=1)
                        for i in range(b):
                            pred_i = pred_b[i].cpu().numpy()
                            tgt_i = target_mask[i].cpu().numpy()
                            num_classes = meta.get("num_classes", int(max(tgt_i.max(), pred_i.max()) + 1))
                            sample_id = meta.get("sample_id", f"sample_{global_idx:04d}")
                            fig = plot_target_vs_prediction(
                                tgt_i, pred_i, num_classes=num_classes, title=f"Epoch {epoch} - {sample_id}"
                            )
                            out_path = epoch_img_dir / f"{sample_id}.png"
                            fig.savefig(out_path, dpi=100, bbox_inches="tight")

                            # Add to TensorBoard
                            if writer is not None:
                                fig.canvas.draw()
                                img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
                                img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                                img_t = torch.from_numpy(img_arr).float() / 255.0
                                writer.add_image(f"val/epoch_{epoch}/{sample_id}", img_t, epoch, dataformats="HWC")

                            plt.close(fig)
                            global_idx += 1
                model.train()

        # Update epoch progress bar postfix
        postfix = {"loss": f"{np.mean(epoch_losses):.4f}"}
        if val_miou:
            postfix["val_miou"] = f"{val_miou[-1]:.3f}"
        epoch_pbar.set_postfix(**postfix)

    if writer is not None:
        writer.flush()
        writer.close()

    return TrainHistory(
        losses=losses,
        acc=accs,
        miou=mious,
        val_losses=val_losses,
        val_acc=val_acc,
        val_miou=val_miou,
    )

