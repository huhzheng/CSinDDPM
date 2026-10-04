"""Explicit 3D pore connectivity metrics for 6-, 18-, or 26-neighbour rules."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .io import as_pore_mask


_CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}
_AXES = {"z": 0, "y": 1, "x": 2}


def _labels_on_face(labels: np.ndarray, axis: int, index: int) -> set[int]:
    face = np.take(labels, index, axis=axis)
    return {int(value) for value in np.unique(face) if value != 0}


def connectivity_metrics(volume, connectivity: int = 6) -> dict:
    """Measure largest-component and directional spanning pore fractions.

    Fractions with suffix ``_total`` use all voxels as denominator; fractions
    with suffix ``_pore`` use pore voxels.  Direction names assume array order
    ``(z, y, x)`` and are reported explicitly to avoid silent axis swaps.
    """

    if connectivity not in _CONNECTIVITY_RANK:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    pore = as_pore_mask(volume)
    structure = ndimage.generate_binary_structure(3, _CONNECTIVITY_RANK[connectivity])
    labels, component_count = ndimage.label(pore, structure=structure)
    counts = np.bincount(labels.ravel())
    if counts.size:
        counts[0] = 0
    pore_voxels = int(pore.sum())
    total_voxels = int(pore.size)
    largest = int(counts.max()) if pore_voxels else 0

    directional = {}
    all_spanning_labels = set()
    for name, axis in _AXES.items():
        low = _labels_on_face(labels, axis, 0)
        high = _labels_on_face(labels, axis, -1)
        spanning_labels = sorted(low.intersection(high))
        all_spanning_labels.update(spanning_labels)
        spanning_voxels = int(counts[spanning_labels].sum()) if spanning_labels else 0
        directional[name] = {
            "array_axis": axis,
            "spanning_component_count": len(spanning_labels),
            "spanning_pore_voxels": spanning_voxels,
            "spanning_porosity_total": spanning_voxels / total_voxels,
            "spanning_fraction_pore": (
                spanning_voxels / pore_voxels if pore_voxels else 0.0
            ),
            "percolates": bool(spanning_labels),
        }

    percolating_voxels = (
        int(counts[sorted(all_spanning_labels)].sum()) if all_spanning_labels else 0
    )
    isolated_voxels = max(0, pore_voxels - largest)

    return {
        "axis_order": "zyx",
        "neighbour_connectivity": connectivity,
        "total_voxels": total_voxels,
        "pore_voxels": pore_voxels,
        "total_porosity": pore_voxels / total_voxels,
        "component_count": int(component_count),
        "largest_component_voxels": largest,
        "largest_component_porosity_total": largest / total_voxels,
        "largest_component_fraction_pore": largest / pore_voxels if pore_voxels else 0.0,
        "connected_definition": "largest_component",
        "connected_pore_fraction": largest / total_voxels,
        "isolated_pore_fraction": isolated_voxels / total_voxels,
        "percolating_definition": "union_of_xyz_face_spanning_components",
        "percolating_fraction": percolating_voxels / total_voxels,
        "percolation_x": directional["x"]["percolates"],
        "percolation_y": directional["y"]["percolates"],
        "percolation_z": directional["z"]["percolates"],
        "directional_spanning": directional,
    }
