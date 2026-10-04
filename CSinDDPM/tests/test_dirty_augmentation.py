"""Regression tests for the submitted dirty-volume algorithm."""

import math

import pytest
import torch

from demo_diffusion.augmentation import apply_dirty_cuboid


def test_dirty_probability_zero_and_one():
    volume = torch.ones((1, 1, 16, 16, 16))
    clean, clean_meta = apply_dirty_cuboid(volume, 0.0, 0.1, 0.3)
    assert not clean_meta["dirty_applied"]
    assert set(torch.unique(clean).tolist()) == {1.0}

    dirty, dirty_meta = apply_dirty_cuboid(
        volume,
        1.0,
        0.1,
        0.3,
        generator=torch.Generator().manual_seed(7),
    )
    assert dirty_meta["dirty_applied"]
    assert dirty_meta["dirty_mode"] in {"pore", "grain"}
    assert set(torch.unique(dirty).tolist()).issubset({-1.0, 1.0})


def test_dirty_probability_is_about_thirty_percent():
    generator = torch.Generator().manual_seed(2026)
    volume = torch.ones((1, 1, 8, 8, 8))
    applied = 0
    pore_modes = 0
    trials = 10000
    for _ in range(trials):
        _, metadata = apply_dirty_cuboid(
            volume, 0.30, 0.10, 0.30, generator=generator
        )
        applied += int(metadata["dirty_applied"])
        pore_modes += int(metadata["dirty_mode"] == "pore")
    observed = applied / trials
    assert 0.28 <= observed <= 0.32
    assert 0.47 <= pore_modes / applied <= 0.53


def test_fixed_seed_reproduces_volume_and_metadata():
    volume = torch.cat(
        [torch.ones(1, 1, 8, 8, 8), -torch.ones(1, 1, 8, 8, 8)], dim=2
    )
    first, first_meta = apply_dirty_cuboid(
        volume, 1.0, 0.1, 0.3, generator=torch.Generator().manual_seed(99)
    )
    second, second_meta = apply_dirty_cuboid(
        volume, 1.0, 0.1, 0.3, generator=torch.Generator().manual_seed(99)
    )
    assert torch.equal(first, second)
    assert first_meta == second_meta


def test_cuboid_fractions_and_porosity_metadata_are_exact():
    volume = torch.cat(
        [torch.ones(1, 1, 32, 64, 64), -torch.ones(1, 1, 32, 64, 64)],
        dim=2,
    )
    augmented, metadata = apply_dirty_cuboid(
        volume,
        1.0,
        0.10,
        0.30,
        generator=torch.Generator().manual_seed(11),
    )
    assert metadata["porosity_before"] == pytest.approx(0.5)
    assert metadata["porosity_after"] == pytest.approx(
        float((augmented > 0).float().mean())
    )
    bbox = metadata["cuboid_bbox"]
    for (start, end), length in zip(bbox, volume.shape[-3:]):
        size = end - start
        assert math.ceil(length * 0.10) <= size <= math.floor(length * 0.30)
        assert 0 <= start < end <= length


def test_invalid_augmentation_parameters_fail_loudly():
    volume = torch.ones((1, 1, 8, 8, 8))
    with pytest.raises(ValueError):
        apply_dirty_cuboid(volume, 1.1, 0.1, 0.3)
    with pytest.raises(ValueError):
        apply_dirty_cuboid(volume, 0.3, 0.4, 0.3)
