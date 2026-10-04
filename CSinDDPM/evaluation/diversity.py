"""Within-method diversity metrics over the complete fixed sample set."""

from __future__ import annotations

import itertools

import numpy as np

from .dd import difference_degree
from .io import as_pore_mask
from .mpc import directional_mpc


def pairwise_hamming_diversity(volumes) -> dict:
    masks = [as_pore_mask(volume) for volume in volumes]
    if len(masks) < 2:
        raise ValueError("at least two volumes are required")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("all volumes must have the same shape")
    distances = [
        float(np.not_equal(masks[i], masks[j]).mean())
        for i, j in itertools.combinations(range(len(masks)), 2)
    ]
    values = np.asarray(distances, dtype=np.float64)
    return {
        "sample_count": len(masks),
        "pair_count": len(distances),
        "mean_pairwise_hamming": float(values.mean()),
        "sd_pairwise_hamming": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min_pairwise_hamming": float(values.min()),
        "max_pairwise_hamming": float(values.max()),
    }


def pairwise_structural_diversity(volumes, mpc_n_points=2, max_lag=None) -> dict:
    """Pairwise Hamming, porosity, and Eq.-8 MPC differences."""

    masks = [as_pore_mask(volume) for volume in volumes]
    base = pairwise_hamming_diversity(masks)
    curves = [directional_mpc(mask, mpc_n_points, max_lag)["curves"] for mask in masks]
    porosities = [float(mask.mean()) for mask in masks]
    porosity_differences = []
    dd_values = {axis: [] for axis in "xyz"}
    for i, j in itertools.combinations(range(len(masks)), 2):
        porosity_differences.append(abs(porosities[i] - porosities[j]))
        for axis in "xyz":
            dd_values[axis].append(
                difference_degree(curves[i][axis], curves[j][axis])
            )
    base.update(
        {
            "mean_pairwise_porosity_difference": float(
                np.mean(porosity_differences)
            ),
            "mean_pairwise_mpc_dd_x": float(np.mean(dd_values["x"])),
            "mean_pairwise_mpc_dd_y": float(np.mean(dd_values["y"])),
            "mean_pairwise_mpc_dd_z": float(np.mean(dd_values["z"])),
            "mpc_definition_id": "manuscript_eq8_literal_v1",
        }
    )
    return base
