"""Porosity metrics under the explicit pore-mask convention."""

from __future__ import annotations

from .io import as_pore_mask


def porosity(volume) -> float:
    """Fraction of voxels marked as pore."""

    return float(as_pore_mask(volume).mean())


def porosity_error(volume, target: float) -> dict:
    if not 0.0 <= target <= 1.0:
        raise ValueError("target porosity must be in [0, 1]")
    observed = porosity(volume)
    return {
        "target_porosity": float(target),
        "generated_porosity": observed,
        "signed_error": observed - float(target),
        "absolute_error": abs(observed - float(target)),
    }
