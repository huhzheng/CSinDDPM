import numpy as np
import pytest
import tifffile
import torch

from demo_diffusion.demo_image_datasets import (
    ImageDataset,
    binary_mask,
    compute_porosity,
    read_binary_tiff,
)


@pytest.fixture
def binary_tiff(tmp_path):
    volume = np.zeros((64, 64, 64), dtype=np.uint8)
    volume[:16] = 255
    path = tmp_path / "binary_volume.tif"
    tifffile.imwrite(path, volume)
    return path


def test_tiff_phase_encoding_and_porosity(binary_tiff):
    encoded = read_binary_tiff(binary_tiff)
    assert encoded.shape == (64, 64, 64)
    assert set(encoded.reshape(-1).tolist()) == {-1.0, 1.0}
    tensor = torch.from_numpy(encoded).unsqueeze(0).unsqueeze(0)
    assert float(compute_porosity(tensor)) == pytest.approx(0.25)


def test_resizing_keeps_binary_values(binary_tiff):
    dataset = ImageDataset(
        64,
        binary_tiff,
        scale_init=0.75,
        repeats=1,
    )
    assert dataset.volume.shape == (1, 48, 48, 48)
    assert set(torch.unique(dataset.volume).tolist()) == {-1.0, 1.0}


def test_constant_standard_binary_volumes_remain_valid():
    zeros = binary_mask(np.zeros((4, 4, 4)), pore_is_high=True)
    highs = binary_mask(np.ones((4, 4, 4)) * 255, pore_is_high=True)
    assert not zeros.any()
    assert highs.all()
