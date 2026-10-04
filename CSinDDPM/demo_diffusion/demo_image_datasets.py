"""Single-volume TIFF loading with an explicit binary pore/grain convention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

from .augmentation import GRAIN_VALUE, PORE_VALUE
from .mpi_compat import COMM_WORLD


Number = Union[int, float]


def infer_binary_threshold(volume: np.ndarray) -> float:
    """Return the midpoint of the observed low/high values."""

    minimum = float(np.min(volume))
    maximum = float(np.max(volume))
    if minimum == maximum:
        raise ValueError("cannot infer a binary threshold from a constant volume")
    return (minimum + maximum) / 2.0


def binary_mask(
    volume: np.ndarray,
    threshold: Optional[Number] = None,
    pore_is_high: bool = True,
) -> np.ndarray:
    """Convert a numeric 3D volume to an explicit Boolean pore mask."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D volume, received shape {array.shape}")
    if threshold is None and float(np.min(array)) == float(np.max(array)):
        value = float(array.flat[0])
        if value not in {-1.0, 0.0, 1.0, 255.0}:
            raise ValueError(
                "constant non-standard volume requires an explicit threshold"
            )
        high_phase = np.full(array.shape, value > 0, dtype=bool)
        return high_phase if pore_is_high else ~high_phase
    cutoff = infer_binary_threshold(array) if threshold is None else float(threshold)
    return array > cutoff if pore_is_high else array <= cutoff


def encode_binary_volume(
    volume: np.ndarray,
    threshold: Optional[Number] = None,
    pore_is_high: bool = True,
) -> np.ndarray:
    """Encode pore as +1 and grain as -1 using float32 values only."""

    mask = binary_mask(volume, threshold=threshold, pore_is_high=pore_is_high)
    return np.where(mask, PORE_VALUE, GRAIN_VALUE).astype(np.float32)


def read_binary_tiff(
    path: Union[str, Path],
    threshold: Optional[Number] = None,
    pore_is_high: bool = True,
) -> np.ndarray:
    """Read and strictly encode one TIFF volume."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"training TIFF not found: {source}")
    return encode_binary_volume(
        tifffile.imread(source),
        threshold=threshold,
        pore_is_high=pore_is_high,
    )


def compute_porosity(x: torch.Tensor) -> torch.Tensor:
    """Compute per-volume porosity from the +1 pore convention."""

    if x.ndim < 3:
        raise ValueError("porosity requires at least three spatial dimensions")
    return (x > 0).float().mean(dim=(-1, -2, -3))


def _parse_threshold(value):
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    scale_init=1.0,
    scale_factor=0.75,
    stop_scale=0,
    current_scale=0,
    binary_threshold=None,
    pore_is_high=True,
    num_workers=0,
    seed=0,
):
    """Create an infinite loader over one explicitly encoded 3D volume."""

    del class_cond, random_crop, random_flip
    if not data_dir:
        raise ValueError("--data_dir must identify a TIFF volume")
    dataset = ImageDataset(
        image_size,
        data_dir,
        shard=COMM_WORLD.Get_rank(),
        num_shards=COMM_WORLD.Get_size(),
        scale_init=scale_init,
        scale_factor=scale_factor,
        stop_scale=stop_scale,
        current_scale=current_scale,
        binary_threshold=_parse_threshold(binary_threshold),
        pore_is_high=pore_is_high,
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
    )
    while True:
        yield from loader


class ImageDataset(Dataset):
    """A repeated single-volume dataset used by CSinDDPM training."""

    def __init__(
        self,
        resolution,
        image_path,
        shard=0,
        num_shards=1,
        scale_init=1.0,
        scale_factor=0.75,
        stop_scale=0,
        current_scale=0,
        binary_threshold=None,
        pore_is_high=True,
        repeats=10000,
    ):
        super().__init__()
        del shard, num_shards
        self.resolution = int(resolution)
        self.repeats = int(repeats)
        encoded = read_binary_tiff(
            image_path,
            threshold=binary_threshold,
            pore_is_high=pore_is_high,
        )

        current_factor = scale_init * math.pow(
            scale_factor, stop_scale - current_scale
        )
        target_shape = tuple(max(1, round(dim * current_factor)) for dim in encoded.shape)
        if target_shape != encoded.shape:
            mask = encoded > 0
            factors = tuple(
                target / original for target, original in zip(target_shape, encoded.shape)
            )
            mask = zoom(mask.astype(np.uint8), factors, order=0) > 0
            encoded = np.where(mask, PORE_VALUE, GRAIN_VALUE).astype(np.float32)

        self.volume = torch.from_numpy(encoded).unsqueeze(0)
        unique_values = set(float(value) for value in torch.unique(self.volume).tolist())
        if not unique_values.issubset({GRAIN_VALUE, PORE_VALUE}):
            raise RuntimeError(f"non-binary values after preprocessing: {unique_values}")
        self.porosity = compute_porosity(self.volume)

    def __len__(self):
        return self.repeats

    def __getitem__(self, idx):
        del idx
        return self.volume.clone(), self.porosity.clone()
