"""Literal implementation of manuscript Eq. (8), kept definition-versioned."""

from __future__ import annotations

import numpy as np

from .io import as_pore_mask


_AXES = {"z": 0, "y": 1, "x": 2}


def equation8_mpc_curve(
    volume,
    axis: int,
    n_points: int = 2,
    max_lag: int | None = None,
) -> np.ndarray:
    """Compute E[product_j S(u + j*h)] for Boolean pore state S.

    This is the literal code interpretation of Eq. (8).  The historical plot
    values cannot be regenerated from the submitted repository, and their
    near-0.9 first values are inconsistent with Eq. (8) for volumes whose pore
    fractions are near 0.2.  Results therefore carry the definition id
    ``manuscript_eq8_literal_v1`` and must not silently replace the old table.
    """

    pore = as_pore_mask(volume)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    largest_lag = (pore.shape[axis] - 1) // (n_points - 1)
    if max_lag is None:
        max_lag = largest_lag
    if not 0 <= max_lag <= largest_lag:
        raise ValueError(f"max_lag must be within [0, {largest_lag}]")

    curve = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        usable = pore.shape[axis] - (n_points - 1) * lag
        product = np.ones(
            tuple(usable if dim == axis else size for dim, size in enumerate(pore.shape)),
            dtype=bool,
        )
        for point in range(n_points):
            slices = [slice(None)] * 3
            start = point * lag
            slices[axis] = slice(start, start + usable)
            product &= pore[tuple(slices)]
        curve[lag] = product.mean()
    return curve


def directional_mpc(volume, n_points: int = 2, max_lag: int | None = None) -> dict:
    return {
        "definition_id": "manuscript_eq8_literal_v1",
        "axis_order": "zyx",
        "n_points": int(n_points),
        "curves": {
            name: equation8_mpc_curve(volume, axis, n_points, max_lag).tolist()
            for name, axis in _AXES.items()
        },
    }
