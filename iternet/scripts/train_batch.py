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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from iternet.config import GridConfig, ModelConfig, TrainConfig
from iternet.data_discovery import discover_train_test
from iternet.dataset import IternetDataset, collate_batch, collate_single
from iternet.model import IternetUNet
from iternet.preprocessing import preprocess_pair
from iternet.train import TrainHistory, train_segmentation


def _save_loss_iou_curves(
    history: TrainHistory,
    out_dir: Path,
    batches_per_epoch: int = 1,
) -> None:
    """Save loss and regression metric curves for train and val to PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = len(history.losses)
    if steps == 0:
        return

    n_epochs = (steps + batches_per_epoch - 1) // batches_per_epoch
    train_epoch_loss = []
    train_epoch_rmse = []
    for e in range(n_epochs):
        start = e * batches_per_epoch
        end = min(start + batches_per_epoch, steps)
        if start < end:
            train_epoch_loss.append(float(np.mean(history.losses[start:end])))
            train_epoch_rmse.append(float(np.mean(history.rmse[start:end])))

    fig: Any
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

    ax2.plot(ep, train_epoch_rmse, "b-", alpha=0.8, label="train")
    if history.val_rmse:
        ax2.plot(np.arange(len(history.val_rmse)), history.val_rmse, "g-", label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.legend()
    ax2.set_title("RMSE (train & val)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_iou_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--nx", type=int, default=600)
    parser.add_argument("--nz", type=int, default=300)
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

    # Processed mode: expects 2-channel input matrices (*.npz matrix_data)
    model_cfg = ModelConfig()

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

    if model_cfg.in_channels != 2:
        raise SystemExit("This training script expects processed 2-channel input (.npz matrix_data).")
    model = IternetUNet(
        in_channels=model_cfg.in_channels,
        patch_size=model_cfg.patch_size,
        base_channels=model_cfg.base_channels,
        depth=model_cfg.depth,
        blocks_per_stage=model_cfg.blocks_per_stage,
        stem_blocks=model_cfg.stem_blocks,
        out_channels=model_cfg.out_channels,
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
        "model_in_channels": model_cfg.in_channels,
        "model_patch_size": model_cfg.patch_size,
        "model_base_channels": model_cfg.base_channels,
        "model_depth": model_cfg.depth,
        "model_blocks_per_stage": model_cfg.blocks_per_stage,
        "model_stem_blocks": model_cfg.stem_blocks,
        "model_out_channels": model_cfg.out_channels,
        "mse_weight": TrainConfig.mse_weight,
        "mae_weight": TrainConfig.mae_weight,
        "boundary_weight_factor": TrainConfig.boundary_weight_factor,
        "boundary_weight_radius": TrainConfig.boundary_weight_radius,
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
        config_dict=config_dict,
        mse_weight=TrainConfig.mse_weight,
        mae_weight=TrainConfig.mae_weight,
        boundary_weight_factor=TrainConfig.boundary_weight_factor,
        boundary_weight_radius=TrainConfig.boundary_weight_radius,
        boundary_loss_weight=TrainConfig.boundary_loss_weight,
    )

    print(f"Train samples: {len(train_pairs)}, Test samples: {len(test_pairs)}")
    print(f"Final train loss: {history.losses[-1]:.4f}, rmse: {history.rmse[-1]:.4f}")
    if history.val_losses:
        print(f"Final val loss: {history.val_losses[-1]:.4f}, val rmse: {history.val_rmse[-1]:.4f}")

    batches_per_epoch = len(train_loader)
    _save_loss_iou_curves(history, log_dir, batches_per_epoch=batches_per_epoch)

    # Save model
    ckpt_path = log_dir / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "arch": "patch_unet_poly_abcd",
            "in_channels": model_cfg.in_channels,
            "patch_size": model_cfg.patch_size,
            "base_channels": model_cfg.base_channels,
            "depth": model_cfg.depth,
            "blocks_per_stage": model_cfg.blocks_per_stage,
            "stem_blocks": model_cfg.stem_blocks,
            "out_channels": model_cfg.out_channels,
        },
        ckpt_path,
    )
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
