from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from iternet.io import parse_ie2d_res
from iternet.preprocessing import PreprocessResult, preprocess_pair


@dataclass(frozen=True)
class SamplePaths:
    """Pair of files for one training sample."""

    ie2d_res: Path
    target_matrix: Path


class IternetDataset(Dataset):
    """
    Dataset for (ie2d_res.dat, target_matrix.npz) pairs.

    Each item returns:
    - meas_values_01: (1578,)
    - target_matrix_norm: (Z, X)
    - meta: dict
    """

    def __init__(
        self,
        samples: list[SamplePaths],
        *,
        nx: int,
        nz: int,
        grid_overrides: dict | None = None,
        value_kind: str = "auto",
        current_a: float = 1.0,
    ) -> None:
        self.samples = samples
        self.nx = nx
        self.nz = nz
        self.grid_overrides = grid_overrides or {}
        self.value_kind = value_kind
        self.current_a = current_a

        self._cache: list[PreprocessResult | None] = [None for _ in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sp = self.samples[idx]
        cached = self._cache[idx]
        if cached is None:
            ie2d = parse_ie2d_res(sp.ie2d_res)
            cached = preprocess_pair(
                ie2d=ie2d,
                target_matrix_path=sp.target_matrix,
                nx=self.nx,
                nz=self.nz,
                grid_overrides=self.grid_overrides,
                value_kind=self.value_kind,
                current_a=self.current_a,
            )
            self._cache[idx] = cached

        meta = {
            "x_coords": cached.x_coords,
            "z_coords": cached.z_coords,
            "meas_stats": cached.meas_stats,
            "target_stats": cached.target_stats,
            "target_matrix_raw": cached.target_matrix_raw,
            "sample_id": sp.target_matrix.stem,
        }

        return cached.meas_values_01, cached.target_matrix_norm, meta


def collate_single(batch):
    """Collate function for batch_size=1 (keeps variable-length measurements)."""
    if len(batch) != 1:
        raise ValueError("collate_single expects batch_size=1")
    meas_values_01, target_matrix_norm, meta = batch[0]
    return (
        meas_values_01.unsqueeze(0),  # (1, 1578)
        target_matrix_norm.unsqueeze(0),  # (1, Z, X)
        meta,
    )


def collate_batch(batch):
    """
    Collate for batch_size > 1: stack meas vectors and target matrices.
    """
    meas_values_01 = torch.stack([b[0] for b in batch], dim=0)  # (B, 1578)
    target_matrix_norm = torch.stack([b[1] for b in batch], dim=0)  # (B, Z, X)
    meta = batch[0][2]
    return meas_values_01, target_matrix_norm, meta

