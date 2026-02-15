from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from iternet.io.ie2 import IE2Model
from iternet.io.ie2d import IE2DResData
from iternet.io.ie2d import IE2DMeasurement


@dataclass(frozen=True)
class PreprocessResult:
    """Single training sample tensors."""

    # Measurement tokens: (N_meas, F)
    meas_tokens: torch.Tensor
    # Query coords: (N_grid, 2) in normalized [-1,1]
    grid_xy: torch.Tensor
    # Target mask: (Z, X) with class ids
    target_mask: torch.Tensor

    # Grid for visualization / reconstruction
    x_coords: np.ndarray
    z_coords: np.ndarray

    # Metadata
    num_classes: int
    class_rho: dict[int, float]
    value_kind: str


def _grid_from_ie2(model: IE2Model, *, nx: int, nz: int, overrides: dict[str, float | None]) -> tuple[np.ndarray, np.ndarray]:
    if overrides.get("x_min") is not None and overrides.get("x_max") is not None:
        x_min = float(overrides["x_min"])
        x_max = float(overrides["x_max"])
    else:
        # IMPORTANT: prefer true model extents from points.
        # 'Box' in IE2 is often a plotting window and may truncate depth.
        xs = [x for x, _ in model.points_xz.values()]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))

    if overrides.get("z_min") is not None and overrides.get("z_max") is not None:
        z_min = float(overrides["z_min"])
        z_max = float(overrides["z_max"])
    else:
        # IMPORTANT: prefer true model extents from points (see note above).
        zs = [z for _, z in model.points_xz.values()]
        z_min, z_max = float(np.min(zs)), float(np.max(zs))

    x_coords = np.linspace(x_min, x_max, nx, dtype=np.float32)
    z_coords = np.linspace(z_min, z_max, nz, dtype=np.float32)
    return x_coords, z_coords


def rasterize_ie2_model(
    model: IE2Model,
    *,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    background_class: int = 0,
) -> tuple[np.ndarray, dict[int, float]]:
    """
    Rasterize polygon bodies into a (Z,X) mask.

    - Uses body.color as class id by default
    - Later bodies override earlier ones (simple painter's algorithm)
    """

    nz = len(z_coords)
    nx = len(x_coords)
    mask = np.full((nz, nx), background_class, dtype=np.int64)

    # Precompute grid points (X,Z) flattened
    xx, zz = np.meshgrid(x_coords, z_coords)
    pts = np.stack([xx.reshape(-1), zz.reshape(-1)], axis=1)

    class_rho: dict[int, float] = {}

    for body in model.bodies:
        poly = [model.points_xz[i] for i in body.point_indices if i in model.points_xz]
        if len(poly) < 3:
            continue
        path = MplPath(np.array(poly, dtype=np.float32), closed=True)
        inside = path.contains_points(pts).reshape(nz, nx)
        mask[inside] = int(body.color)
        class_rho[int(body.color)] = float(body.rho)

    return mask, class_rho


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
    ie2: IE2Model,
    nx: int,
    nz: int,
    grid_overrides: dict[str, Any] | None = None,
    value_kind: str = "auto",
    current_a: float = 1.0,
) -> PreprocessResult:
    """
    Convert (measurements, model) into tensors for training/inference.
    """

    grid_overrides = grid_overrides or {}
    x_coords, z_coords = _grid_from_ie2(ie2, nx=nx, nz=nz, overrides=grid_overrides)
    mask, class_rho = rasterize_ie2_model(ie2, x_coords=x_coords, z_coords=z_coords)

    num_classes = int(max([0, *class_rho.keys()]) + 1)

    tokens = build_measurement_tokens(
        ie2d,
        x_min=float(x_coords.min()),
        x_max=float(x_coords.max()),
        z_min=float(z_coords.min()),
        z_max=float(z_coords.max()),
        value_kind=value_kind,
        current_a=current_a,
    )
    grid_q = build_grid_queries(x_coords=x_coords, z_coords=z_coords)

    return PreprocessResult(
        meas_tokens=torch.from_numpy(tokens),
        grid_xy=torch.from_numpy(grid_q),
        target_mask=torch.from_numpy(mask),
        x_coords=x_coords,
        z_coords=z_coords,
        num_classes=num_classes,
        class_rho=class_rho,
        value_kind=value_kind,
    )

