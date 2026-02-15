"""
Batch training script with validation and test image saves.

Usage:
  python -m iternet.scripts.train_batch [options]

Or from project root:
  python -m iternet.scripts.train_batch --data_dir data/processed --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from iternet.config import GridConfig, ModelConfig, TrainConfig
from iternet.data_discovery import discover_train_test
from iternet.dataset import IternetDataset, collate_batch, collate_single
from iternet.model import IternetPerceiver
from iternet.preprocessing import preprocess_pair
from iternet.train import TrainHistory, train_segmentation


def _save_loss_iou_curves(
    history: TrainHistory,
    out_dir: Path,
    batches_per_epoch: int = 1,
) -> None:
    """Save loss and IoU curves for train and val to PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = len(history.losses)
    if steps == 0:
        return

    n_epochs = (steps + batches_per_epoch - 1) // batches_per_epoch
    train_epoch_loss = []
    train_epoch_miou = []
    for e in range(n_epochs):
        start = e * batches_per_epoch
        end = min(start + batches_per_epoch, steps)
        if start < end:
            train_epoch_loss.append(float(np.mean(history.losses[start:end])))
            train_epoch_miou.append(float(np.mean(history.miou[start:end])))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ep = np.arange(len(train_epoch_loss))
    ax1.plot(ep, train_epoch_loss, "b-", alpha=0.8, label="train")
    if history.val_losses:
        ax1.plot(np.arange(len(history.val_losses)), history.val_losses, "g-", label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss (train & val)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, train_epoch_miou, "b-", alpha=0.8, label="train")
    if history.val_miou:
        ax2.plot(np.arange(len(history.val_miou)), history.val_miou, "g-", label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.legend()
    ax2.set_title("IoU (train & val)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_iou_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _infer_num_classes_and_in_features(train_samples: list, grid_cfg: GridConfig) -> tuple[int, int]:
    """Infer from samples: max num_classes across all, in_features from first."""
    from iternet.io import parse_ie2_model, parse_ie2d_res

    grid_overrides = {
        "x_min": grid_cfg.x_min,
        "x_max": grid_cfg.x_max,
        "z_min": grid_cfg.z_min,
        "z_max": grid_cfg.z_max,
    }
    max_classes = 0
    max_in_mask = 0
    in_features = 0
    for sp in train_samples:
        ie2d = parse_ie2d_res(sp.ie2d_res)
        ie2 = parse_ie2_model(sp.ie2_model)
        prep = preprocess_pair(
            ie2d=ie2d,
            ie2=ie2,
            nx=grid_cfg.look_nx,
            nz=grid_cfg.look_nz,
            grid_overrides=grid_overrides,
        )
        max_classes = max(max_classes, prep.num_classes)
        max_in_mask = max(max_in_mask, int(prep.target_mask.max()))
        if in_features == 0:
            in_features = int(prep.meas_tokens.shape[1])
    # Ensure model covers all class ids in data (target values 0..max_in_mask)
    num_classes = max(max_classes, max_in_mask + 1, 2)
    return num_classes, in_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Iternet on batch data with validation")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Base dir with train/ and test/ subdirs",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=Path, default=Path("iternet/runs"))
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--x_min", type=float, default=-300.0)
    parser.add_argument("--x_max", type=float, default=300.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=150.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_pairs, test_pairs = discover_train_test(data_dir)
    if not train_pairs:
        raise SystemExit(f"No train pairs found in {data_dir}")
    if not test_pairs:
        print("Warning: no test pairs found, validation will be skipped")

    grid_cfg = GridConfig(
        look_nx=args.nx,
        look_nz=args.nz,
        x_min=args.x_min,
        x_max=args.x_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    grid_overrides = {
        "x_min": grid_cfg.x_min,
        "x_max": grid_cfg.x_max,
        "z_min": grid_cfg.z_min,
        "z_max": grid_cfg.z_max,
    }

    num_classes, in_features = _infer_num_classes_and_in_features(train_pairs, grid_cfg)
    model_cfg = ModelConfig(num_classes=num_classes)

    train_ds = IternetDataset(
        samples=train_pairs,
        nx=grid_cfg.look_nx,
        nz=grid_cfg.look_nz,
        grid_overrides=grid_overrides,
    )
    test_ds = IternetDataset(
        samples=test_pairs,
        nx=grid_cfg.look_nx,
        nz=grid_cfg.look_nz,
        grid_overrides=grid_overrides,
    )

    # Train: shuffle=True — каждый эпоху сэмплы в случайном порядке
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch if args.batch_size > 1 else collate_single,
        num_workers=0,
    )
    # Test: shuffle=False — прогон один раз, фиксированный порядок
    val_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
        num_workers=0,
    ) if test_pairs else None

    model = IternetPerceiver(
        in_features=in_features,
        token_dim=model_cfg.token_dim,
        latent_dim=model_cfg.latent_dim,
        num_latents=model_cfg.num_latents,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        num_classes=num_classes,
        dropout=model_cfg.dropout,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "nx": args.nx,
        "nz": args.nz,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "boundary_weight_factor": TrainConfig.boundary_weight_factor,
        "boundary_weight_radius": TrainConfig.boundary_weight_radius,
        "ce_weight": TrainConfig.ce_weight,
        "dice_weight": TrainConfig.dice_weight,
        "boundary_loss_weight": TrainConfig.boundary_loss_weight,
    }

    history = train_segmentation(
        model=model,
        loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=TrainConfig.weight_decay,
        device=args.device,
        log_dir=log_dir,
        val_loader=val_loader,
        boundary_weight_factor=TrainConfig.boundary_weight_factor,
        boundary_weight_radius=TrainConfig.boundary_weight_radius,
        ce_weight=TrainConfig.ce_weight,
        dice_weight=TrainConfig.dice_weight,
        boundary_loss_weight=TrainConfig.boundary_loss_weight,
        config_dict=config_dict,
    )

    print(f"Train samples: {len(train_pairs)}, Test samples: {len(test_pairs)}")
    print(f"Final train loss: {history.losses[-1]:.4f}, miou: {history.miou[-1]:.4f}")
    if history.val_losses:
        print(f"Final val loss: {history.val_losses[-1]:.4f}, val miou: {history.val_miou[-1]:.4f}")

    batches_per_epoch = len(train_loader)
    _save_loss_iou_curves(history, log_dir, batches_per_epoch=batches_per_epoch)

    # Save model
    ckpt_path = log_dir / "model.pt"
    torch.save({"model": model.state_dict(), "num_classes": num_classes, "in_features": in_features}, ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
