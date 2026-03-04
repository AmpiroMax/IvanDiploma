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
    Dataset for input/target pairs.

    Supported input formats:
    - `.dat` (ERT measurements) — legacy mode
    - `.npz` with `matrix_data` of shape (H, W, 2) — new processed mode

    Each item returns:
    - input_tensor_01: (1578,) or (2, H, W)
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
            if sp.ie2d_res.suffix.lower() == ".dat":
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
            elif sp.ie2d_res.suffix.lower() == ".npz":
                cached = preprocess_pair(
                    input_matrix_path=sp.ie2d_res,
                    target_matrix_path=sp.target_matrix,
                    nx=self.nx,
                    nz=self.nz,
                    grid_overrides=self.grid_overrides,
                )
            else:
                raise ValueError(f"Unsupported input file type: {sp.ie2d_res}")
            self._cache[idx] = cached

        meta = {
            "x_coords": cached.x_coords,
            "z_coords": cached.z_coords,
            "input_kind": cached.input_kind,
            "input_stats": cached.input_stats,
            "input_tensor_raw": cached.input_tensor_raw,
            "target_stats": cached.target_stats,
            "target_matrix_raw": cached.target_matrix_raw,
            "sample_id": sp.target_matrix.stem,
        }

        return cached.input_tensor_01, cached.target_matrix_norm, meta


def collate_single(batch):
    """Collate function for batch_size=1."""
    if len(batch) != 1:
        raise ValueError("collate_single expects batch_size=1")
    input_tensor_01, target_matrix_norm, meta = batch[0]
    return (
        input_tensor_01.unsqueeze(0),
        target_matrix_norm.unsqueeze(0),  # (1, Z, X)
        meta,
    )


def collate_batch(batch):
    """
    Collate for batch_size > 1: stack inputs and target matrices.
    """
    input_tensor_01 = torch.stack([b[0] for b in batch], dim=0)
    target_matrix_norm = torch.stack([b[1] for b in batch], dim=0)  # (B, Z, X)
    meta = batch[0][2]
    return input_tensor_01, target_matrix_norm, meta

