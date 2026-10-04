"""Difference degree from manuscript Eq. (9)."""

from __future__ import annotations

import numpy as np


def difference_degree(reference_curve, reconstructed_curve) -> float:
    """Return the unnormalised sum of squared MPC-curve differences."""

    reference = np.asarray(reference_curve, dtype=np.float64)
    reconstruction = np.asarray(reconstructed_curve, dtype=np.float64)
    if reference.ndim != 1 or reconstruction.ndim != 1:
        raise ValueError("both curves must be one-dimensional")
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"curve shapes differ: {reference.shape} vs {reconstruction.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(reconstruction).all():
        raise ValueError("curves contain non-finite values")
    return float(np.square(reference - reconstruction).sum())
