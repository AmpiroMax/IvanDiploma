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


def plot_prediction(
    pred_mask: np.ndarray,
    *,
    title: str = "Predicted mask",
    num_classes: int | None = None,
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> object:
    fig, ax = plt.subplots(figsize=(10, 4))
    n = int(pred_mask.max() + 1) if num_classes is None else int(num_classes)
    cmap, norm = _discrete_cmap(max(n, 1))
    extent = _extent_from_coords(x_coords, z_coords)
    im = ax.imshow(
        pred_mask,
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


def plot_target_vs_prediction(
    target: np.ndarray,
    pred: np.ndarray,
    *,
    num_classes: int | None = None,
    title: str = "Target vs Prediction",
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> object:
    """Side-by-side target and prediction for validation saves."""
    n = int(max(target.max(), pred.max()) + 1) if num_classes is None else int(num_classes)
    cmap, norm = _discrete_cmap(max(n, 1))
    extent = _extent_from_coords(x_coords, z_coords)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, arr, lbl in [(ax1, target, "Target"), (ax2, pred, "Prediction")]:
        ax.imshow(
            arr,
            origin="upper",
            aspect="equal" if extent is not None else "auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            extent=extent,
        )
        ax.set_title(lbl)
        ax.set_xlabel("X" if extent is not None else "X index")
        ax.set_ylabel("Z (depth)" if extent is not None else "Z index")
        ax.grid(True, alpha=0.2)
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

