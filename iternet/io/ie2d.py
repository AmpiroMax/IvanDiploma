from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class IE2DMeasurement:
    """Single ABMN measurement (ERT)."""

    xa: float
    za: float
    xb: float | None
    zb: float | None
    xm: float
    zm: float
    xn: float
    zn: float
    value: float  # app. resistivity or resistance depending on header
    b_infinite: bool = False

    @property
    def has_b(self) -> bool:
        return (self.xb is not None) and (self.zb is not None) and (not self.b_infinite)


@dataclass(frozen=True)
class IE2DResData:
    """Parsed IE2D resistivity-like dataset."""

    sys_path: str
    electrode_spacing: float
    measurement_type: int  # 0=app.resistivity, 1=resistance
    measurements: tuple[IE2DMeasurement, ...]


def _iter_nonempty_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        yield line


def parse_ie2d_res(path: str | Path) -> IE2DResData:
    """
    Parse an IE2D-like .dat file with ABMN + value rows.

    Expected header (based on sample):
    line 1: path to sys file (string)
    line 2: electrode spacing (float)
    line 5-6: 'Type of measurement' label then int
    then a line with number of measurements, then the measurement rows.
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = list(_iter_nonempty_lines(text))
    if len(lines) < 10:
        raise ValueError(f"File too short: {p}")

    sys_path = lines[0]
    electrode_spacing = float(lines[1])

    # Find measurement type label
    mt_idx = None
    for i, line in enumerate(lines):
        if "Type of measurement" in line:
            mt_idx = i
            break
    if mt_idx is None or mt_idx + 1 >= len(lines):
        raise ValueError("Could not find 'Type of measurement' in file header.")
    measurement_type = int(float(lines[mt_idx + 1]))

    # Heuristic: measurement count is the first integer line after measurement_type.
    count_idx = None
    for i in range(mt_idx + 2, min(mt_idx + 30, len(lines))):
        try:
            n = int(float(lines[i]))
        except ValueError:
            continue
        # In sample it is 1953
        if n > 0:
            count_idx = i
            break
    if count_idx is None:
        raise ValueError("Could not locate measurements count in header.")
    n_meas = int(float(lines[count_idx]))

    # Find the first row that looks like a measurement:
    # - 4-electrode: "4 xa za xb zb xm zm xn zn value" (>= 10 tokens)
    # - 3-electrode: "3 xa za xm zm xn zn value" (>= 8 tokens), where B is at infinity
    start_idx = None
    for i in range(count_idx + 1, len(lines)):
        parts = lines[i].split()
        if parts and parts[0] == "4" and len(parts) >= 10:
            start_idx = i
            break
        if parts and parts[0] == "3" and len(parts) >= 8:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find first measurement row.")

    measurements: list[IE2DMeasurement] = []
    for line in lines[start_idx : start_idx + n_meas]:
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]

        if tag == "4" and len(parts) >= 10:
            # 4 xa za xb zb xm zm xn zn value
            xa, za, xb, zb, xm, zm, xn, zn = map(float, parts[1:9])
            value = float(parts[9])
            measurements.append(
                IE2DMeasurement(
                    xa=xa,
                    za=za,
                    xb=xb,
                    zb=zb,
                    xm=xm,
                    zm=zm,
                    xn=xn,
                    zn=zn,
                    value=value,
                    b_infinite=False,
                )
            )
            continue

        if tag == "3" and len(parts) >= 8:
            # 3 xa za xm zm xn zn value  (B at infinity)
            xa, za, xm, zm, xn, zn = map(float, parts[1:7])
            value = float(parts[7])
            measurements.append(
                IE2DMeasurement(
                    xa=xa,
                    za=za,
                    xb=None,
                    zb=None,
                    xm=xm,
                    zm=zm,
                    xn=xn,
                    zn=zn,
                    value=value,
                    b_infinite=True,
                )
            )
            continue

    if not measurements:
        raise ValueError("No measurements parsed.")

    return IE2DResData(
        sys_path=sys_path,
        electrode_spacing=electrode_spacing,
        measurement_type=measurement_type,
        measurements=tuple(measurements),
    )
