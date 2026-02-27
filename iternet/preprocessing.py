from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iternet.io.ie2d import IE2DResData
from iternet.io.ie2d import IE2DMeasurement


@dataclass(frozen=True)
class PreprocessResult:
    """Single training sample tensors for matrix regression."""

    # Measurement values from .dat last column, normalized to [0,1]: (1578,)
    meas_values_01: torch.Tensor
    # Target matrix in normalized domain [-1,1]: (Z, X), float32
    target_matrix_norm: torch.Tensor

    # Grid for visualization / reconstruction
    x_coords: np.ndarray
    z_coords: np.ndarray

    # Metadata
    meas_stats: dict[str, float]
    target_stats: dict[str, float]
    target_matrix_raw: np.ndarray
    value_kind: str


def _grid_from_overrides(*, nx: int, nz: int, overrides: dict[str, float | None]) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(overrides.get("x_min") if overrides.get("x_min") is not None else -300.0)
    x_max = float(overrides.get("x_max") if overrides.get("x_max") is not None else 300.0)
    z_min = float(overrides.get("z_min") if overrides.get("z_min") is not None else 0.0)
    z_max = float(overrides.get("z_max") if overrides.get("z_max") is not None else 150.0)
    x_coords = np.linspace(x_min, x_max, nx, dtype=np.float32)
    z_coords = np.linspace(z_min, z_max, nz, dtype=np.float32)
    return x_coords, z_coords


def load_target_matrix_npz(path: str | Path, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load 2D target matrix from .npz; uses first stored array."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as d:
        if len(d.files) == 0:
            raise ValueError(f"NPZ file has no arrays: {p}")
        arr = np.asarray(d[d.files[0]])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {p}, got shape {arr.shape}")
    if expected_shape is not None and arr.shape != expected_shape:
        if arr.T.shape == expected_shape:
            arr = arr.T
        else:
            raise ValueError(f"Target matrix shape {arr.shape} does not match expected {expected_shape} for {p}")
    return arr.astype(np.float32, copy=False)


def normalize_target_matrix(target_raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """
    Normalize target matrix into [-1,1].

    Uses sign-log compression first to stabilize large amplitudes:
        signed_log = sign(v) * log1p(abs(v))
    then min-max scaling to [-1,1].
    """
    signed_log = np.sign(target_raw) * np.log1p(np.abs(target_raw))
    vmin = float(np.min(signed_log))
    vmax = float(np.max(signed_log))
    denom = max(vmax - vmin, 1e-9)
    target_norm = (2.0 * (signed_log - vmin) / denom - 1.0).astype(np.float32)
    stats = {"signed_log_min": vmin, "signed_log_max": vmax}
    return target_norm, stats


def denormalize_target_matrix(target_norm: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    """Map normalized model output [-1,1] back to original target value range."""
    vmin = float(stats["signed_log_min"])
    vmax = float(stats["signed_log_max"])
    signed_log = ((np.clip(target_norm, -1.0, 1.0) + 1.0) * 0.5) * (vmax - vmin) + vmin
    return (np.sign(signed_log) * (np.expm1(np.abs(signed_log)))).astype(np.float32)


def build_meas_values_vector_01(
    data: IE2DResData,
    *,
    expected_n: int = 1578,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build a fixed-length vector from the last column of each measurement row.

    Returns:
      - values_01: (expected_n,) float32 in [0,1]
      - stats: min/max before normalization
    """
    vals = np.asarray([float(m.value) for m in data.measurements], dtype=np.float32)
    if vals.shape[0] != expected_n:
        raise ValueError(f"Expected {expected_n} measurements, got {vals.shape[0]}")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    denom = max(vmax - vmin, 1e-9)
    vals_01 = ((vals - vmin) / denom).astype(np.float32)
    vals_01 = np.clip(vals_01, 0.0, 1.0)
    stats = {"meas_min": vmin, "meas_max": vmax}
    return vals_01, stats


def _dist(x1: float, z1: float, x2: float, z2: float) -> float:
    return float(np.hypot(x1 - x2, z1 - z2))


def _normal_field_L(m: IE2DMeasurement) -> float:
    """
    Normal field L for I=1, ρ0=1:
      L = 1/(2π) * ( 1/AM - 1/AN - 1/BM + 1/BN )
    If B is at infinity: L = 1/(2π) * (1/AM - 1/AN)
    """
    eps = 1e-9
    am = _dist(m.xa, m.za, m.xm, m.zm)
    an = _dist(m.xa, m.za, m.xn, m.zn)
    am = max(am, eps)
    an = max(an, eps)

    if m.has_b and m.xb is not None and m.zb is not None:
        bm = _dist(m.xb, m.zb, m.xm, m.zm)
        bn = _dist(m.xb, m.zb, m.xn, m.zn)
        bm = max(bm, eps)
        bn = max(bn, eps)
        val = (1.0 / am) - (1.0 / an) - (1.0 / bm) + (1.0 / bn)
    else:
        val = (1.0 / am) - (1.0 / an)

    return float((1.0 / (2.0 * np.pi)) * val)


def apparent_resistivity_from_value(
    data: IE2DResData,
    m: IE2DMeasurement,
    *,
    value_kind: str,
    current_a: float,
) -> float:
    """
    Convert stored value to apparent resistivity ρa.
    """
    vk = value_kind.lower().strip()
    if vk == "auto":
        # Header: 0=app.resistivity, 1=resistance
        vk = "rho_a" if data.measurement_type == 0 else "resistance"

    if vk == "rho_a":
        return float(m.value)

    L = _normal_field_L(m)
    if abs(L) < 1e-12:
        return float("nan")

    if vk == "resistance":
        # R = ΔU / I
        # ρa = K * R, where K = 1/L
        return float((1.0 / L) * float(m.value))

    if vk == "voltage":
        # ρa = ΔU / (L * I)
        I = float(current_a) if float(current_a) != 0 else 1.0
        return float(float(m.value) / (L * I))

    raise ValueError(f"Unknown value_kind: {value_kind}. Use auto|voltage|resistance|rho_a.")


def build_measurement_tokens(
    data: IE2DResData,
    *,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    value_kind: str,
    current_a: float,
) -> np.ndarray:
    """
    Convert ABMN + value measurements into token features.

    Uses log10(value) and multiple derived distances to help generalization.
    """

    eps = 1e-9
    feats: list[list[float]] = []

    def norm(v: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin + eps:
            return 0.0
        return float(2.0 * (v - vmin) / (vmax - vmin) - 1.0)

    for m in data.measurements:
        xa, xm, xn = m.xa, m.xm, m.xn
        za, zm, zn = m.za, m.zm, m.zn

        xb = m.xb if m.has_b else None
        zb = m.zb if m.has_b else None

        x_mid_mn = 0.5 * (xm + xn)
        mn = abs(xm - xn)
        am = abs(xa - xm)
        an = abs(xa - xn)

        # If B is at infinity (3-electrode), AB is undefined.
        # Use a reasonable proxy for "depth/scale": distance from A to MN-midpoint.
        if xb is None:
            ab = abs(xa - x_mid_mn)
            x_mid_ab = xa
            bm = 0.0
            bn = 0.0
            xb_norm = 0.0
            zb_norm = 0.0
            b_inf = 1.0
        else:
            ab = abs(xa - xb)
            x_mid_ab = 0.5 * (xa + xb)
            bm = abs(xb - xm)
            bn = abs(xb - xn)
            xb_norm = norm(xb, x_min, x_max)
            zb_norm = norm(zb if zb is not None else 0.0, z_min, z_max)
            b_inf = 0.0

        rho_a = apparent_resistivity_from_value(data, m, value_kind=value_kind, current_a=current_a)
        if not np.isfinite(rho_a) or rho_a <= 0:
            rho_a = eps
        logv = float(np.log10(max(float(rho_a), eps)))

        feats.append(
            [
                norm(xa, x_min, x_max),
                norm(za, z_min, z_max),
                xb_norm,
                zb_norm,
                norm(xm, x_min, x_max),
                norm(zm, z_min, z_max),
                norm(xn, x_min, x_max),
                norm(zn, z_min, z_max),
                norm(x_mid_ab, x_min, x_max),
                norm(x_mid_mn, x_min, x_max),
                ab,
                mn,
                am,
                an,
                bm,
                bn,
                logv,
                b_inf,
            ]
        )

    arr = np.asarray(feats, dtype=np.float32)

    # Normalize distance features (distance columns + logv; keep coordinates already normalized)
    # Keep coordinate features already normalized.
    dist_cols = slice(10, 16)
    arr[:, dist_cols] = (arr[:, dist_cols] - arr[:, dist_cols].mean(axis=0)) / (arr[:, dist_cols].std(axis=0) + 1e-6)
    # Normalize logv
    arr[:, 16] = (arr[:, 16] - arr[:, 16].mean()) / (arr[:, 16].std() + 1e-6)
    # b_inf flag is arr[:,17] - leave as {0,1}

    return arr


def build_grid_queries(
    *,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
) -> np.ndarray:
    """
    Build (N_grid,2) query coordinates in normalized [-1,1] for decoder.
    """

    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    z_min, z_max = float(z_coords.min()), float(z_coords.max())
    eps = 1e-9

    def norm(v: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax <= vmin + eps:
            return np.zeros_like(v, dtype=np.float32)
        return (2.0 * (v - vmin) / (vmax - vmin) - 1.0).astype(np.float32)

    xx, zz = np.meshgrid(x_coords, z_coords)
    qx = norm(xx, x_min, x_max)
    qz = norm(zz, z_min, z_max)
    q = np.stack([qx.reshape(-1), qz.reshape(-1)], axis=1)
    return q


def preprocess_pair(
    *,
    ie2d: IE2DResData,
    target_matrix_path: str | Path,
    nx: int,
    nz: int,
    grid_overrides: dict[str, Any] | None = None,
    value_kind: str = "auto",
    current_a: float = 1.0,
) -> PreprocessResult:
    """
    Convert (measurements, target matrix) into tensors for training/inference.
    """

    grid_overrides = grid_overrides or {}
    x_coords, z_coords = _grid_from_overrides(nx=nx, nz=nz, overrides=grid_overrides)
    target_raw = load_target_matrix_npz(target_matrix_path, expected_shape=(nz, nx))
    target_norm, target_stats = normalize_target_matrix(target_raw)

    _ = (value_kind, current_a)
    meas_values_01, meas_stats = build_meas_values_vector_01(ie2d, expected_n=1578)

    return PreprocessResult(
        meas_values_01=torch.from_numpy(meas_values_01),
        target_matrix_norm=torch.from_numpy(target_norm),
        x_coords=x_coords,
        z_coords=z_coords,
        meas_stats=meas_stats,
        target_stats=target_stats,
        target_matrix_raw=target_raw,
        value_kind=value_kind,
    )

