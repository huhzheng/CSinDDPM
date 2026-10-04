"""Shared binary-volume input conventions for revision metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from demo_diffusion.demo_image_datasets import binary_mask


def as_pore_mask(volume) -> np.ndarray:
    """Return a 3D Boolean array without guessing a phase from signed values."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D volume, received {array.shape}")
    if array.dtype == np.bool_:
        return array.copy()
    unique = np.unique(array)
    if set(unique.tolist()).issubset({-1, 1}):
        return array > 0
    if set(unique.tolist()).issubset({0, 1}):
        return array > 0
    raise ValueError(
        "numeric arrays must already use {-1,+1} or {0,1}; use "
        "load_tiff_mask() to apply an explicit TIFF threshold"
    )


def load_tiff_mask(path, threshold=None, pore_is_high=True) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return binary_mask(
        tifffile.imread(source), threshold=threshold, pore_is_high=pore_is_high
    )
