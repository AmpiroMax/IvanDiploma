from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ExportConfig:
    """Controls polygon extraction and output size."""

    min_area_cells: int = 50
    simplify_step: int = 5  # keep every N-th vertex
    hatch: int = 1
    eta: float = 0.0


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _find_tail_start(template_lines: list[str]) -> int:
    """
    Find the start of the survey config tail that we should keep verbatim.
    In your sample it's the line containing '- step,'.
    """
    for i, ln in enumerate(template_lines):
        if " - step" in ln and "Nsteps" in ln:
            return i
    # Fallback: keep only header if not found
    return len(template_lines)


def _contours_for_class(mask: np.ndarray, class_id: int) -> list[np.ndarray]:
    """
    Extract polygon contours using matplotlib (no skimage dependency).
    Returns list of arrays with shape (N,2) in (row, col) float coordinates.
    """
    binary = (mask == class_id).astype(np.float32)
    if binary.sum() == 0:
        return []

    # Contour at 0.5 to capture boundary between 0 and 1
    fig = plt.figure(figsize=(4, 3))
    try:
        cs = plt.contour(binary, levels=[0.5])
        segs = []
        for seglist in cs.allsegs:
            for seg in seglist:
                if seg.shape[0] >= 3:
                    # seg is (N,2): x=col, y=row in matplotlib; convert to (row,col)
                    rc = np.stack([seg[:, 1], seg[:, 0]], axis=1)
                    segs.append(rc)
        return segs
    finally:
        plt.close(fig)


def _rc_to_xz(rc: np.ndarray, x_coords: np.ndarray, z_coords: np.ndarray) -> np.ndarray:
    """
    Convert (row,col) float coords to (x,z) by nearest index sampling.
    """
    rows = np.clip(np.round(rc[:, 0]).astype(int), 0, len(z_coords) - 1)
    cols = np.clip(np.round(rc[:, 1]).astype(int), 0, len(x_coords) - 1)
    xs = x_coords[cols]
    zs = z_coords[rows]
    return np.stack([xs, zs], axis=1)


def _polygon_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _dedupe_points(points: list[tuple[float, float]], tol: float = 1e-4) -> tuple[list[tuple[float, float]], dict[tuple[int, int], int]]:
    """
    Deduplicate points by rounding; return unique list and lookup (rounded)->index.
    Indices are 1-based for IE2.
    """
    uniq: list[tuple[float, float]] = []
    lookup: dict[tuple[int, int], int] = {}

    for x, z in points:
        key = (int(round(x / tol)), int(round(z / tol)))
        if key in lookup:
            continue
        lookup[key] = len(uniq) + 1
        uniq.append((x, z))

    return uniq, lookup


def export_prediction_to_ie2(
    *,
    pred_mask: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    class_rho: dict[int, float],
    template_ie2_path: str | Path,
    out_path: str | Path,
    config: ExportConfig | None = None,
) -> Path:
    """
    Export predicted mask (Z,X) into an IE2 polygon model file.

    Notes:
    - Uses matplotlib contour extraction on each class id > 0.
    - Saves a simplified polygon set; not guaranteed to match your original editor exactly,
      but should be readable by the same tooling.
    """
    cfg = config or ExportConfig()
    template_path = Path(template_ie2_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    template_lines = _read_lines(template_path)
    tail_start = _find_tail_start(template_lines)
    tail_lines = template_lines[tail_start:] if tail_start < len(template_lines) else []

    # Build bodies as polygons
    bodies: list[dict] = []
    all_pts: list[tuple[float, float]] = []

    class_ids = sorted(int(c) for c in np.unique(pred_mask) if int(c) != 0)
    for cid in class_ids:
        # crude area filter in cell space
        if int((pred_mask == cid).sum()) < cfg.min_area_cells:
            continue

        segs_rc = _contours_for_class(pred_mask, cid)
        for rc in segs_rc:
            xz = _rc_to_xz(rc, x_coords=x_coords, z_coords=z_coords)
            if cfg.simplify_step > 1:
                xz = xz[:: cfg.simplify_step]
            if xz.shape[0] < 3:
                continue
            # ensure closed for area check
            if not np.allclose(xz[0], xz[-1]):
                xz = np.vstack([xz, xz[0]])
            area = _polygon_area(xz)
            if area <= 0:
                continue

            pts_list = [(float(x), float(z)) for x, z in xz[:-1]]  # exclude duplicate closing point
            bodies.append(
                {
                    "rho": float(class_rho.get(cid, 100.0)),
                    "eta": float(cfg.eta),
                    "color": int(cid),
                    "hatch": int(cfg.hatch),
                    "points": pts_list,
                }
            )
            all_pts.extend(pts_list)

    # Deduplicate global points
    uniq_pts, lookup = _dedupe_points(all_pts)

    # Assign indices per body
    for body in bodies:
        idxs = []
        for x, z in body["points"]:
            key = (int(round(x / 1e-4)), int(round(z / 1e-4)))
            idxs.append(lookup[key])
        body["idxs"] = idxs

    nbodies = len(bodies)
    npoints = len(uniq_pts)

    # Header: keep title/array type from template if possible
    title = template_lines[0] if template_lines else "Predicted model"
    array_type = template_lines[1] if len(template_lines) > 1 else "Amn+mnB"

    # Build file lines
    out_lines: list[str] = []
    out_lines.append(f"{title} (predicted)")
    out_lines.append(array_type)
    out_lines.append(f" {nbodies} {npoints} 0  1.5 -1 - Nbodies,Npoints,IPkey,PreciseFactor, SaveKey XZ-elem.")

    # Bodies
    for i, body in enumerate(bodies, start=1):
        pts = body["idxs"]
        out_lines.append(
            f"{body['rho']:.2f}   {body['eta']:.2f} {len(pts)} {body['color']} {body['hatch']}  -   {i} body: - Rho,Eta,Npoints,color,hatch"
        )
        out_lines.append(" ".join(str(p) for p in pts) + "  - points numbers")

    # Points list
    for idx, (x, z) in enumerate(uniq_pts, start=1):
        out_lines.append(f"{x:8.3f}  {z:8.3f} -  {idx} - th point X,Z")

    # Append tail from template (survey configuration)
    if tail_lines:
        out_lines.extend(tail_lines)

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_path

