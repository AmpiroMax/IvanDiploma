from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IE2Body:
    """Single polygonal body in IE2 model."""

    rho: float
    eta: float
    n_points: int
    color: int
    hatch: int
    point_indices: tuple[int, ...]  # 1-based indices


@dataclass(frozen=True)
class IE2Model:
    """Parsed IE2 geometry model."""

    title: str
    nbodies: int
    npoints: int
    bodies: tuple[IE2Body, ...]
    points_xz: dict[int, tuple[float, float]]  # 1-based -> (x,z)
    # Box / extents (if present)
    box: tuple[float, float, float, float] | None = None  # (xmin,xmax,zmin,zmax)


def _clean(line: str) -> str:
    # Keep parsing robust to alignment / repeated spaces
    return line.strip().replace("\t", " ")


def parse_ie2_model(path: str | Path) -> IE2Model:
    """
    Parse a simplified IE2 file variant (as in the provided sample).

    The sample looks like:
    - title line
    - second line: array type
    - third line: "Nbodies,Npoints,..." (starts with nbodies npoints)
    - then for each body:
        "<Rho> <Eta> <Npoints> <color> <hatch> - comment"
        "<point indices...> - points numbers"
    - then list of points:
        "<x> <z> - <idx> - th point X,Z"
    - then survey config and at end a "Box:" line with extents
    """

    p = Path(path)
    lines = [ln for ln in (_clean(x) for x in p.read_text(encoding="utf-8", errors="ignore").splitlines()) if ln]
    if len(lines) < 10:
        raise ValueError(f"File too short: {p}")

    title = lines[0]

    # Find line with Nbodies,Npoints comment, or fallback to line 2 (new format)
    header_idx = None
    for i, ln in enumerate(lines[:20]):
        if "Nbodies" in ln and "Npoints" in ln:
            header_idx = i
            break
    if header_idx is None and len(lines) >= 3:
        # New format: third line is " nbodies npoints ..."
        header_idx = 2
    if header_idx is None:
        raise ValueError("Could not find Nbodies/Npoints header line.")

    header_nums = lines[header_idx].split()
    nbodies = int(float(header_nums[0]))
    npoints = int(float(header_nums[1]))

    # Parse bodies
    bodies: list[IE2Body] = []
    cursor = header_idx + 1
    for _ in range(nbodies):
        # body line: Rho Eta Npoints color hatch ...
        parts = lines[cursor].split()
        rho = float(parts[0])
        eta = float(parts[1])
        b_npoints = int(float(parts[2]))
        color = int(float(parts[3]))
        hatch = int(float(parts[4]))
        cursor += 1

        # indices line
        idx_parts = [x for x in lines[cursor].replace("-", " ").split() if x.strip().isdigit()]
        point_indices = tuple(int(x) for x in idx_parts[:b_npoints])
        cursor += 1

        bodies.append(
            IE2Body(
                rho=rho,
                eta=eta,
                n_points=b_npoints,
                color=color,
                hatch=hatch,
                point_indices=point_indices,
            )
        )

    # Parse points: look for lines like "<x> <z> - <idx> - th point" or new format "<x> <z> - <idx>"
    points_xz: dict[int, tuple[float, float]] = {}
    for i in range(cursor, len(lines)):
        ln = lines[i]
        # New format: "x z - idx" (no "th point")
        if " - " in ln:
            left, right = ln.split(" - ", 1)
            left_parts = left.split()
            right_parts = right.split()
            if len(left_parts) >= 2 and right_parts:
                try:
                    x = float(left_parts[0])
                    z = float(left_parts[1])
                    idx = int(float(right_parts[0]))
                    points_xz[idx] = (x, z)
                except (ValueError, IndexError):
                    pass
            continue
        # Legacy format with "th point" or "point X,Z"
        if "th point" in ln or "point X,Z" in ln:
            parts = ln.replace("-", " ").split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[0])
                z = float(parts[1])
            except ValueError:
                continue
            idx = None
            for token in parts[2:]:
                try:
                    v = int(float(token))
                except ValueError:
                    continue
                idx = v
                break
            if idx is not None:
                points_xz[idx] = (x, z)

    # Parse box extents if present
    box = None
    for ln in lines[::-1]:
        if "Box:" in ln:
            parts = ln.split("Box:")[0].split()
            # ... Xmin Xmax ScaleX Zmin Zmax ScaleZ
            if len(parts) >= 6:
                xmin = float(parts[0])
                xmax = float(parts[1])
                zmin = float(parts[3])
                zmax = float(parts[4])
                box = (xmin, xmax, zmin, zmax)
            break

    if len(points_xz) == 0:
        raise ValueError("No points parsed from .ie2 file.")

    return IE2Model(
        title=title,
        nbodies=nbodies,
        npoints=npoints,
        bodies=tuple(bodies),
        points_xz=points_xz,
        box=box,
    )

