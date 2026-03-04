"""
Discover input/target pairs from folder structure.

Expected layout:
Legacy:
  train/electrical_resistivity_tomography/*.dat
  train/models/*.npz
  test/electrical_resistivity_tomography/*.dat
  test/models/*.npz

Processed (current):
  train/input/*.npz
  train/output/*.npz
  test/input/*.npz
  test/output/*.npz

Pairs are matched by stem (e.g. 224.dat <-> 224.npz).
"""

from __future__ import annotations

from pathlib import Path

from iternet.dataset import SamplePaths


def discover_pairs(
    input_dir: Path,
    target_dir: Path,
    *,
    input_glob: str = "*.dat",
    target_glob: str = "*.npz",
) -> list[SamplePaths]:
    """
    Find all (input, target) pairs where stem matches.
    Returns only pairs where both files exist.
    """
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)

    input_files = {f.stem: f for f in input_dir.glob(input_glob)}
    target_files = {f.stem: f for f in target_dir.glob(target_glob)}

    common = sorted(set(input_files) & set(target_files))
    return [
        SamplePaths(ie2d_res=input_files[s], target_matrix=target_files[s])
        for s in common
    ]


def discover_train_test(
    base_dir: Path,
    *,
    # Legacy layout
    train_ert: str = "train/electrical_resistivity_tomography",
    train_models: str = "train/models",
    test_ert: str = "test/electrical_resistivity_tomography",
    test_models: str = "test/models",
    # Processed layout
    train_input: str = "train/input",
    train_output: str = "train/output",
    test_input: str = "test/input",
    test_output: str = "test/output",
) -> tuple[list[SamplePaths], list[SamplePaths]]:
    """
    Discover train and test pairs from a base directory.
    """
    base = Path(base_dir)
    if (base / train_input).exists() and (base / train_output).exists():
        train = discover_pairs(base / train_input, base / train_output, input_glob="*.npz", target_glob="*.npz")
        test = discover_pairs(base / test_input, base / test_output, input_glob="*.npz", target_glob="*.npz")
    else:
        train = discover_pairs(base / train_ert, base / train_models, input_glob="*.dat", target_glob="*.npz")
        test = discover_pairs(base / test_ert, base / test_models, input_glob="*.dat", target_glob="*.npz")
    return train, test
