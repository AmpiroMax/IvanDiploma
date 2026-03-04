from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import numpy as np

from iternet.io.ie2d import IE2DResData
from iternet.preprocessing import apparent_resistivity_from_value


@dataclass(frozen=True)
class Figures:
    mask_fig: object | None = None
    meas_fig: object | None = None
    pred_fig: object | None = None
    rho_fig: object | None = None


def _discrete_cmap(num_classes: int, base: str = "tab20") -> tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    base_cmap = plt.get_cmap(base)
    colors = [base_cmap(i % base_cmap.N) for i in range(num_classes)]
    cmap = mcolors.ListedColormap(colors, name=f"{base}_{num_classes}")
    boundaries = np.arange(-0.5, num_classes + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _extent_from_coords(x_coords: np.ndarray | None, z_coords: np.ndarray | None) -> list[float] | None:
    if x_coords is None or z_coords is None:
        return None
    # Z axis is shown downward, consistent with IE2 geometry plotting.
    return [float(x_coords.min()), float(x_coords.max()), float(z_coords.max()), float(z_coords.min())]


def plot_mask(
    mask: np.ndarray,
    *,
    title: str = "Target mask",
    num_classes: int | None = None,
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> object:
    fig, ax = plt.subplots(figsize=(10, 4))
    n = int(mask.max() + 1) if num_classes is None else int(num_classes)
    cmap, norm = _discrete_cmap(max(n, 1))
    extent = _extent_from_coords(x_coords, z_coords)
    im = ax.imshow(
        mask,
        origin="upper",
        aspect="equal" if extent is not None else "auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=extent,
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, ticks=list(range(max(n, 1))))
    ax.set_xlabel("X" if extent is not None else "X index")
    ax.set_ylabel("Z (depth)" if extent is not None else "Z index")
    fig.tight_layout()
    return fig


def plot_measurements_tokens(tokens: np.ndarray, *, title: str = "Measurement tokens (debug view)") -> object:
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(tokens.T, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Feature index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_two_channel_image(
    x: np.ndarray,
    *,
    title: str = "Input (2-channel)",
    channel_names: tuple[str, str] = ("ch0", "ch1"),
) -> object:
    """
    Visualize a 2-channel input as:
    - channel 0 (grayscale)
    - channel 1 (grayscale)
    - combined RG image (R=ch0, G=ch1)

    Accepts shapes (2, H, W) or (H, W, 2).
    """
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[0] == 2:
        ch0, ch1 = arr[0], arr[1]
    elif arr.shape[-1] == 2:
        ch0, ch1 = arr[..., 0], arr[..., 1]
    else:
        raise ValueError(f"Expected 2-channel array, got shape {arr.shape}")

    def to01(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32, copy=False)
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))
        denom = max(vmax - vmin, 1e-9)
        return np.clip((a - vmin) / denom, 0.0, 1.0)

    ch0_01 = to01(ch0)
    ch1_01 = to01(ch1)
    rgb = np.zeros((ch0_01.shape[0], ch0_01.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = ch0_01
    rgb[..., 1] = ch1_01

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(ch0_01, origin="upper", cmap="gray", interpolation="nearest")
    ax0.set_title(channel_names[0])
    fig.colorbar(im0, ax=ax0, shrink=0.8)

    im1 = ax1.imshow(ch1_01, origin="upper", cmap="gray", interpolation="nearest")
    ax1.set_title(channel_names[1])
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2.imshow(rgb, origin="upper", interpolation="nearest")
    ax2.set_title("Combined (R=ch0, G=ch1)")

    for ax in axes:
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        ax.grid(True, alpha=0.15)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_prediction(
    pred_matrix: np.ndarray,
    *,
    title: str = "Predicted matrix",
    num_classes: int | None = None,  # kept for backward compatibility
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> object:
    _ = num_classes
    fig, ax = plt.subplots(figsize=(10, 4))
    extent = _extent_from_coords(x_coords, z_coords)
    vmin = float(np.nanmin(pred_matrix))
    vmax = float(np.nanmax(pred_matrix))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    im = ax.imshow(
        pred_matrix,
        origin="upper",
        aspect="equal" if extent is not None else "auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=extent,
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Value")
    ax.set_xlabel("X" if extent is not None else "X index")
    ax.set_ylabel("Z (depth)" if extent is not None else "Z index")
    fig.tight_layout()
    return fig


def plot_target_vs_prediction(
    target: np.ndarray,
    pred: np.ndarray,
    *,
    num_classes: int | None = None,  # kept for backward compatibility
    title: str = "Target vs Prediction",
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> object:
    """Side-by-side target and prediction for validation saves."""
    _ = num_classes
    extent = _extent_from_coords(x_coords, z_coords)
    vmin = float(np.nanmin([np.nanmin(target), np.nanmin(pred)]))
    vmax = float(np.nanmax([np.nanmax(target), np.nanmax(pred)]))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, arr, lbl in [(ax1, target, "Target"), (ax2, pred, "Prediction")]:
        im = ax.imshow(
            arr,
            origin="upper",
            aspect="equal" if extent is not None else "auto",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=extent,
        )
        ax.set_title(lbl)
        ax.set_xlabel("X" if extent is not None else "X index")
        ax.set_ylabel("Z (depth)" if extent is not None else "Z index")
        ax.grid(True, alpha=0.2)
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Value")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _robust_limits(a: np.ndarray, *, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(a, lo))
    vmax = float(np.percentile(a, hi))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_abcd_components(
    *,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    title: str = "A/B/C/D components",
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> object:
    """
    Visualize polynomial decomposition for a single sample.

    Args:
        a,b,c,d: (Z, X) coefficient maps in:
            y = A*1000 + B*100 + C*10 + D
        d: (Z, X) delta D
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)
    if not (a.shape == b.shape == c.shape == d.shape) or a.ndim != 2:
        raise ValueError(f"Expected a,b,c,d all (Z,X), got {a.shape}, {b.shape}, {c.shape}, {d.shape}")

    extent = _extent_from_coords(x_coords, z_coords)

    avmin, avmax = _robust_limits(a)
    bvmin, bvmax = _robust_limits(b)
    cvmin, cvmax = _robust_limits(c)
    dvmin, dvmax = _robust_limits(d)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axa, axb, axc, axd = axes

    ima = axa.imshow(
        a,
        origin="upper",
        aspect="equal" if extent is not None else "auto",
        cmap="viridis",
        vmin=avmin,
        vmax=avmax,
        interpolation="nearest",
        extent=extent,
    )
    axa.set_title("A (×1000)")
    fig.colorbar(ima, ax=axa, shrink=0.85)

    imb = axb.imshow(
        b,
        origin="upper",
        aspect="equal" if extent is not None else "auto",
        cmap="viridis",
        vmin=bvmin,
        vmax=bvmax,
        interpolation="nearest",
        extent=extent,
    )
    axb.set_title("B (×100)")
    fig.colorbar(imb, ax=axb, shrink=0.85)

    imc = axc.imshow(
        c,
        origin="upper",
        aspect="equal" if extent is not None else "auto",
        cmap="viridis",
        vmin=cvmin,
        vmax=cvmax,
        interpolation="nearest",
        extent=extent,
    )
    axc.set_title("C (×10)")
    fig.colorbar(imc, ax=axc, shrink=0.85)

    imd = axd.imshow(
        d,
        origin="upper",
        aspect="equal" if extent is not None else "auto",
        cmap="viridis",
        vmin=dvmin,
        vmax=dvmax,
        interpolation="nearest",
        extent=extent,
    )
    axd.set_title("D (×1)")
    fig.colorbar(imd, ax=axd, shrink=0.85)

    for ax in axes:
        ax.set_xlabel("X" if extent is not None else "X index")
        ax.set_ylabel("Z (depth)" if extent is not None else "Z index")
        ax.grid(True, alpha=0.15)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_true_resistivity(
    *,
    mask: np.ndarray,
    class_rho: dict[int, float],
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    title: str = "True resistivity ρ(x,z) from IE2 bodies",
    alpha_mask: float = 0.25,
) -> object:
    """
    Visualize the target model in true coordinates:
    - background: log10(ρ) image from body Rho values
    - overlay: discrete class mask (semi-transparent)
    """

    rho = np.full_like(mask, np.nan, dtype=np.float32)
    for cid, rv in class_rho.items():
        rho[mask == cid] = float(rv)

    # if some areas are missing (e.g., background class), set to min rho for display
    finite = np.isfinite(rho) & (rho > 0)
    if not np.any(finite):
        rho[:, :] = 1.0
        finite = rho > 0
    min_rho = float(np.nanmin(rho[finite]))
    rho[~finite] = min_rho

    log_rho = np.log10(np.maximum(rho, 1e-9))

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [float(x_coords.min()), float(x_coords.max()), float(z_coords.max()), float(z_coords.min())]
    im = ax.imshow(log_rho, cmap="turbo", origin="upper", aspect="auto", extent=extent, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log10(ρ)")

    # overlay mask
    n = int(mask.max() + 1)
    cmap, norm = _discrete_cmap(max(n, 1))
    ax.imshow(mask, origin="upper", aspect="auto", extent=extent, cmap=cmap, norm=norm, interpolation="nearest", alpha=alpha_mask)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z (depth)")
    fig.tight_layout()
    return fig


def plot_pseudosection(
    data: IE2DResData,
    *,
    title: str = "Pseudo cross-section (from ABMN)",
    depth_scale: float = 0.5,
    levels: int = 20,
    value_kind: str = "auto",
    current_a: float = 1.0,
) -> object:
    """
    Plot a Res2Dinv-like pseudosection:
    x = midpoint of MN
    z_pseudo = depth_scale * |A-B|
    value = measurement value (log10 for stability)

    Note: 'pseudo depth' is a visualization convention, not true depth.
    """
    xs = []
    zs = []
    vs = []
    for m in data.measurements:
        x_mid_mn = 0.5 * (m.xm + m.xn)
        if m.has_b and m.xb is not None:
            scale = abs(m.xa - m.xb)
        else:
            # 3-electrode (B at infinity): use A-to-MN-midpoint as a proxy
            scale = abs(m.xa - x_mid_mn)
        z_pseudo = depth_scale * scale
        xs.append(x_mid_mn)
        zs.append(z_pseudo)
        rho_a = apparent_resistivity_from_value(data, m, value_kind=value_kind, current_a=current_a)
        vs.append(np.log10(max(float(rho_a), 1e-9)))

    xs = np.asarray(xs, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)
    vs = np.asarray(vs, dtype=np.float32)

    tri = mtri.Triangulation(xs, zs)
    fig, ax = plt.subplots(figsize=(10, 4))
    # "cells" look (closer to typical ERT software) – no smooth contour interpolation
    cntr = ax.tripcolor(tri, vs, shading="flat", cmap="turbo")
    ax.invert_yaxis()  # depth goes down
    ax.set_title(title)
    ax.set_xlabel("X (midpoint MN)")
    ax.set_ylabel("Pseudo depth")
    fig.colorbar(cntr, ax=ax, shrink=0.8, label="log10(ρa)")
    fig.tight_layout()
    return fig

