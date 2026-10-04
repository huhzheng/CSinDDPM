"""Seed-level summaries and uncertainty intervals without fabricated replication."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_sd_ci(values, confidence=0.95) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size > 1 and sd > 0:
        sem = stats.sem(array)
        low, high = stats.t.interval(confidence, array.size - 1, loc=mean, scale=sem)
    else:
        low = high = mean
    return {
        "n": int(array.size),
        "mean": mean,
        "sd": sd,
        "confidence": float(confidence),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def bootstrap_mean_ci(values, confidence=0.95, resamples=10000, seed=2026) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "seed": int(seed),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }
