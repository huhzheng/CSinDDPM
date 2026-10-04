"""Physically explicit local augmentation for a single binary 3D volume."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


PORE_VALUE = 1.0
GRAIN_VALUE = -1.0


def _random_scalar(generator: Optional[torch.Generator]) -> float:
    device = generator.device if generator is not None else torch.device("cpu")
    return float(torch.rand((), generator=generator, device=device).item())


def _randint_inclusive(
    low: int, high: int, generator: Optional[torch.Generator]
) -> int:
    if high < low:
        raise ValueError(f"invalid integer interval [{low}, {high}]")
    device = generator.device if generator is not None else torch.device("cpu")
    return int(
        torch.randint(
            low,
            high + 1,
            (),
            generator=generator,
            device=device,
        ).item()
    )


def _size_bounds(length: int, min_frac: float, max_frac: float) -> Tuple[int, int]:
    minimum = max(1, int(torch.ceil(torch.tensor(length * min_frac)).item()))
    maximum = min(length, int(torch.floor(torch.tensor(length * max_frac)).item()))
    if maximum < minimum:
        maximum = minimum
    return minimum, maximum


def apply_dirty_cuboid(
    x: torch.Tensor,
    probability: float,
    min_frac: float,
    max_frac: float,
    pore_value: float = PORE_VALUE,
    grain_value: float = GRAIN_VALUE,
    generator: Optional[torch.Generator] = None,
):
    """Optionally replace one random 3D cuboid by pore or grain voxels.

    The random event is sampled once per call. The same cuboid is therefore
    applied to every leading batch/channel index, while the final three axes
    are interpreted as depth, height, and width. A fresh tensor is returned;
    the input is never modified in place.
    """

    if x.ndim < 3:
        raise ValueError("x must have at least three spatial dimensions")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 < min_frac <= max_frac <= 1.0:
        raise ValueError("fractions must satisfy 0 < min_frac <= max_frac <= 1")
    if pore_value == grain_value:
        raise ValueError("pore_value and grain_value must be different")

    dtype = x.dtype if x.is_floating_point() else torch.float32
    pore = torch.as_tensor(pore_value, dtype=dtype, device=x.device)
    grain = torch.as_tensor(grain_value, dtype=dtype, device=x.device)
    x_aug = torch.where(x > 0, pore, grain)
    porosity_before = float((x_aug > 0).float().mean().item())

    metadata: Dict[str, object] = {
        "dirty_applied": False,
        "dirty_mode": "none",
        "cuboid_bbox": None,
        "porosity_before": porosity_before,
        "porosity_after": porosity_before,
    }
    if _random_scalar(generator) >= probability:
        return x_aug, metadata

    depth, height, width = (int(value) for value in x_aug.shape[-3:])
    sizes = []
    starts = []
    for length in (depth, height, width):
        lower, upper = _size_bounds(length, min_frac, max_frac)
        size = _randint_inclusive(lower, upper, generator)
        start = _randint_inclusive(0, length - size, generator)
        sizes.append(size)
        starts.append(start)
    ends = [start + size for start, size in zip(starts, sizes)]

    use_pore = _random_scalar(generator) < 0.5
    replacement = pore if use_pore else grain
    d0, h0, w0 = starts
    d1, h1, w1 = ends
    x_aug[..., d0:d1, h0:h1, w0:w1] = replacement

    metadata.update(
        {
            "dirty_applied": True,
            "dirty_mode": "pore" if use_pore else "grain",
            "cuboid_bbox": [[d0, d1], [h0, h1], [w0, w1]],
            "porosity_after": float((x_aug > 0).float().mean().item()),
        }
    )
    return x_aug, metadata
