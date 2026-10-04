"""Calibration summaries for requested versus generated porosity."""

from __future__ import annotations

import numpy as np
from scipy import stats


def calibration_summary(targets, observed, training_porosity=None, local_radius=0.03) -> dict:
    target = np.asarray(targets, dtype=np.float64)
    result = np.asarray(observed, dtype=np.float64)
    if target.shape != result.shape or target.ndim != 1 or target.size < 2:
        raise ValueError("targets and observed must be same-length vectors with n >= 2")
    if not (np.isfinite(target).all() and np.isfinite(result).all()):
        raise ValueError("targets and observed must be finite")
    if np.unique(target).size < 2:
        raise ValueError("calibration requires at least two distinct target values")
    slope, intercept, r_value, p_value, stderr = stats.linregress(target, result)
    rho, rho_p = stats.spearmanr(target, result)
    groups = []
    for value in np.unique(target):
        group = result[target == value]
        row = {
            "target": float(value),
            "n": int(group.size),
            "mean_generated": float(group.mean()),
            "sd_generated": float(group.std(ddof=1)) if group.size > 1 else 0.0,
            "bias": float(group.mean() - value),
            "mae": float(np.abs(group - value).mean()),
        }
        if training_porosity is not None:
            row["within_local_training_range"] = bool(
                abs(value - training_porosity) <= local_radius
            )
        groups.append(row)
    return {
        "n": int(target.size),
        "mae": float(np.abs(result - target).mean()),
        "rmse": float(np.sqrt(np.square(result - target).mean())),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_r_squared": float(r_value ** 2),
        "linear_p_value": float(p_value),
        "linear_slope_stderr": float(stderr),
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "training_porosity": (
            None if training_porosity is None else float(training_porosity)
        ),
        "local_radius": float(local_radius),
        "by_target": groups,
    }
