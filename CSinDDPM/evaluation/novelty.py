"""Patch-nearest-neighbour checks for single-volume memorisation risk."""

from __future__ import annotations

import numpy as np

from .io import as_pore_mask


def direct_similarity_metrics(training_volume, generated_volume) -> dict:
    """Voxel-aligned binary similarity; no registration is performed."""

    training = as_pore_mask(training_volume)
    generated = as_pore_mask(generated_volume)
    if training.shape != generated.shape:
        raise ValueError("training and generated volumes must have the same shape")
    intersection = int(np.logical_and(training, generated).sum())
    union = int(np.logical_or(training, generated).sum())
    pore_sum = int(training.sum() + generated.sum())
    agreement = float(np.equal(training, generated).mean())
    return {
        "voxel_agreement": agreement,
        "binary_dice": (2.0 * intersection / pore_sum) if pore_sum else 1.0,
        "binary_iou": (intersection / union) if union else 1.0,
        "normalized_hamming_distance": 1.0 - agreement,
    }


def max_shifted_similarity(training_volume, generated_volume, max_shift=4) -> dict:
    """Find the best periodic integer shift by voxel agreement.

    Periodic rolling avoids size-dependent overlap denominators. The selected
    convention and search radius are returned so this number remains auditable.
    """

    training = as_pore_mask(training_volume)
    generated = as_pore_mask(generated_volume)
    if training.shape != generated.shape:
        raise ValueError("training and generated volumes must have the same shape")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    best_similarity = -1.0
    best_shift = (0, 0, 0)
    for z in range(-max_shift, max_shift + 1):
        for y in range(-max_shift, max_shift + 1):
            for x in range(-max_shift, max_shift + 1):
                shifted = np.roll(generated, shift=(z, y, x), axis=(0, 1, 2))
                similarity = float(np.equal(training, shifted).mean())
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_shift = (z, y, x)
    return {
        "shift_convention": "periodic_roll_zyx",
        "max_shift_searched": int(max_shift),
        "max_shifted_similarity": best_similarity,
        "best_shift_z": best_shift[0],
        "best_shift_y": best_shift[1],
        "best_shift_x": best_shift[2],
    }


def _sample_patches(mask, patch_size, count, rng):
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * 3
    patch_size = tuple(int(value) for value in patch_size)
    if any(value <= 0 for value in patch_size):
        raise ValueError("patch dimensions must be positive")
    if any(patch > size for patch, size in zip(patch_size, mask.shape)):
        raise ValueError(f"patch {patch_size} is larger than volume {mask.shape}")
    starts = [
        rng.integers(0, size - patch + 1, size=count)
        for size, patch in zip(mask.shape, patch_size)
    ]
    return np.stack(
        [
            mask[
                d : d + patch_size[0],
                h : h + patch_size[1],
                w : w + patch_size[2],
            ].reshape(-1)
            for d, h, w in zip(*starts)
        ]
    )


def _nearest_hamming(query, reference, chunk_size=16):
    nearest = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        distances = np.not_equal(chunk[:, None, :], reference[None, :, :]).mean(axis=2)
        nearest[start : start + len(chunk)] = distances.min(axis=1)
    return nearest


def patch_novelty_metrics(
    training_volume,
    generated_volume,
    patch_size=8,
    training_patch_count=512,
    generated_patch_count=128,
    seed=2026,
) -> dict:
    """Compare generated patches to sampled training patches via Hamming distance."""

    training = as_pore_mask(training_volume)
    generated = as_pore_mask(generated_volume)
    rng = np.random.default_rng(seed)
    train_patches = _sample_patches(training, patch_size, training_patch_count, rng)
    generated_patches = _sample_patches(
        generated, patch_size, generated_patch_count, rng
    )
    nearest = _nearest_hamming(generated_patches, train_patches)
    return {
        "metric": "sampled_patch_nearest_normalized_hamming",
        "seed": int(seed),
        "patch_size": ([patch_size] * 3 if isinstance(patch_size, int) else list(patch_size)),
        "training_patch_count": int(training_patch_count),
        "generated_patch_count": int(generated_patch_count),
        "mean_nearest_distance": float(nearest.mean()),
        "sd_nearest_distance": float(nearest.std(ddof=1)) if len(nearest) > 1 else 0.0,
        "mean_nearest_agreement": float((1.0 - nearest).mean()),
        "median_nearest_distance": float(np.median(nearest)),
        "p05_nearest_distance": float(np.quantile(nearest, 0.05)),
        "p95_nearest_distance": float(np.quantile(nearest, 0.95)),
        "minimum_nearest_distance": float(nearest.min()),
        "exact_patch_match_rate": float((nearest == 0).mean()),
    }
