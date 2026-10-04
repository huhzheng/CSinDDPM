"""CPU-only correctness smoke test; it is not generation-quality evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile
import torch
from torch import nn

from demo_diffusion import demo_gaussian_diffusion as gd
from demo_diffusion.augmentation import apply_dirty_cuboid
from demo_diffusion.demo_image_datasets import read_binary_tiff
from evaluation.connectivity import connectivity_metrics


class _ConditionRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0
        self.labels_ok = True

    def forward(self, x, timesteps, labels=None):
        self.calls += 1
        self.labels_ok &= labels is not None and labels.shape == (x.shape[0], 1)
        return torch.zeros_like(x)


def main():
    root = ROOT
    report = {"scope": "functional CPU smoke; not a quality/generalization result"}
    report["datasets"] = {}
    with TemporaryDirectory(prefix="csinddpm_smoke_") as temp_dir:
        datasets = {}
        for name, pore_slices in (("synthetic_25pct", 4), ("synthetic_50pct", 8)):
            path = Path(temp_dir) / f"{name}.tif"
            source = np.zeros((16, 16, 16), dtype=np.uint8)
            source[:pore_slices] = 255
            tifffile.imwrite(path, source)
            datasets[name] = path
        for name, path in datasets.items():
            encoded = read_binary_tiff(path)
            report["datasets"][name] = {
                "shape": list(encoded.shape),
                "porosity": float((encoded > 0).mean()),
                "values": sorted(np.unique(encoded).tolist()),
                "connectivity_6": connectivity_metrics(encoded > 0, 6),
            }

    generator = torch.Generator().manual_seed(2026)
    # Frequency correctness is shape-independent; keep the smoke test fast.
    volume = torch.ones((1, 1, 8, 8, 8), dtype=torch.float32)
    trials = 10000
    applied = 0
    for _ in range(trials):
        augmented, metadata = apply_dirty_cuboid(
            volume, 0.30, 0.10, 0.30, generator=generator
        )
        applied += int(metadata["dirty_applied"])
        if not set(torch.unique(augmented).tolist()).issubset({-1.0, 1.0}):
            raise AssertionError("dirty augmentation created a third phase")
    observed = applied / trials
    if not 0.28 <= observed <= 0.32:
        raise AssertionError(f"dirty frequency {observed} is outside [0.28, 0.32]")
    report["dirty_augmentation"] = {
        "trials": trials,
        "configured_probability": 0.30,
        "observed_probability": observed,
    }

    diffusion = gd.GaussianDiffusion(
        betas=np.array([0.01, 0.02, 0.03], dtype=np.float64),
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
    )
    label = torch.tensor([[0.18]])
    for sampler in ("p_sample_loop", "ddim_sample_loop"):
        model = _ConditionRecorder()
        output = getattr(diffusion, sampler)(
            model,
            (1, 1, 4, 4, 4),
            noise=torch.zeros((1, 1, 4, 4, 4)),
            device=torch.device("cpu"),
            labels=label,
        )
        if output.shape != (1, 1, 4, 4, 4) or not model.labels_ok:
            raise AssertionError(f"conditional propagation failed for {sampler}")
        report[sampler] = {"reverse_calls": model.calls, "shape": list(output.shape)}

    output = root / "revision_outputs/smoke_test/smoke_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {output}")


if __name__ == "__main__":
    main()
