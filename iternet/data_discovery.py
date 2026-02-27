"""
Discover (ERT .dat, target .npz) pairs from folder structure.

Expected layout:
  train/electrical_resistivity_tomography/*.dat
  train/models/*.npz
  test/electrical_resistivity_tomography/*.dat
  test/models/*.npz

Pairs are matched by stem (e.g. 224.dat <-> 224.npz).
"""

from __future__ import annotations

from pathlib import Path

from iternet.dataset import SamplePaths


def discover_pairs(
    ert_dir: Path,
    models_dir: Path,
) -> list[SamplePaths]:
    """
    Find all (dat, npz) pairs where stem matches.
    Returns only pairs where both files exist.
    """
    ert_dir = Path(ert_dir)
    models_dir = Path(models_dir)

    ert_files = {f.stem: f for f in ert_dir.glob("*.dat")}
    model_files = {f.stem: f for f in models_dir.glob("*.npz")}

    common = sorted(set(ert_files) & set(model_files))
    return [
        SamplePaths(ie2d_res=ert_files[s], target_matrix=model_files[s])
        for s in common
    ]


def discover_train_test(
    base_dir: Path,
    *,
    train_ert: str = "train/electrical_resistivity_tomography",
    train_models: str = "train/models",
    test_ert: str = "test/electrical_resistivity_tomography",
    test_models: str = "test/models",
) -> tuple[list[SamplePaths], list[SamplePaths]]:
    """
    Discover train and test pairs from a base directory.
    """
    base = Path(base_dir)
    train = discover_pairs(base / train_ert, base / train_models)
    test = discover_pairs(base / test_ert, base / test_models)
    return train, test
