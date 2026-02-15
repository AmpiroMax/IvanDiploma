from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from iternet.io import parse_ie2_model, parse_ie2d_res
from iternet.preprocessing import PreprocessResult, preprocess_pair


@dataclass(frozen=True)
class SamplePaths:
    """Pair of files for one training sample."""

    ie2d_res: Path
    ie2_model: Path


class IternetDataset(Dataset):
    """
    Dataset for (ie2d_res.dat, model.ie2) pairs.

    Each item returns:
    - meas_tokens: (N_meas, F)
    - grid_xy: (N_grid, 2)
    - target_mask: (Z, X)
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
            ie2 = parse_ie2_model(sp.ie2_model)
            cached = preprocess_pair(
                ie2d=ie2d,
                ie2=ie2,
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
            "num_classes": cached.num_classes,
            "class_rho": cached.class_rho,
            "sample_id": sp.ie2_model.stem,
        }

        return cached.meas_tokens, cached.grid_xy, cached.target_mask, meta


def collate_single(batch):
    """Collate function for batch_size=1 (keeps variable-length measurements)."""
    if len(batch) != 1:
        raise ValueError("collate_single expects batch_size=1")
    meas_tokens, grid_xy, target_mask, meta = batch[0]
    return (
        meas_tokens.unsqueeze(0),  # (1, N, F)
        grid_xy.unsqueeze(0),  # (1, G, 2)
        target_mask.unsqueeze(0),  # (1, Z, X)
        meta,
    )


def collate_batch(batch):
    """
    Collate for batch_size > 1: pad meas_tokens to max N in batch, stack grid_xy and target_mask.
    Assumes fixed grid (nx, nz) so all samples have same grid shape.
    """
    n_meas = [b[0].shape[0] for b in batch]
    max_n = max(n_meas)
    feat_dim = batch[0][0].shape[1]

    meas_list = []
    for i, (mt, gx, tm, meta) in enumerate(batch):
        pad = max_n - mt.shape[0]
        if pad > 0:
            mt = torch.nn.functional.pad(mt, (0, 0, 0, pad), value=0.0)
        meas_list.append(mt)

    meas_tokens = torch.stack(meas_list, dim=0)
    grid_xy = torch.stack([b[1] for b in batch], dim=0)
    target_mask = torch.stack([b[2] for b in batch], dim=0)
    meta = batch[0][3]  # meta from first sample (shared grid shape)

    return meas_tokens, grid_xy, target_mask, meta

