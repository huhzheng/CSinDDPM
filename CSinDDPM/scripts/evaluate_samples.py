"""Evaluate every realization before aggregation and uncertainty estimation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evaluation.connectivity import connectivity_metrics
from evaluation.dd import difference_degree
from evaluation.diversity import (
    pairwise_hamming_diversity,
    pairwise_structural_diversity,
)
from evaluation.io import load_tiff_mask
from evaluation.mpc import directional_mpc
from evaluation.novelty import (
    direct_similarity_metrics,
    max_shifted_similarity,
    patch_novelty_metrics,
)
from evaluation.statistics import mean_sd_ci


def main():
    args = create_parser().parse_args()
    training = load_tiff_mask(
        args.training_tiff, args.binary_threshold, args.pore_is_high
    )
    sample_dir = Path(args.sample_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = sorted(sample_dir.glob(args.pattern))
    if not sample_paths:
        raise FileNotFoundError(f"no samples matched {sample_dir / args.pattern}")
    samples = [
        load_tiff_mask(path, args.binary_threshold, args.pore_is_high)
        for path in sample_paths
    ]
    if any(sample.shape != training.shape for sample in samples):
        raise ValueError("training and generated volumes must share one shape")

    mpc_enabled = args.mpc_definition == "eq8_literal"
    reference_mpc = (
        directional_mpc(training, args.mpc_n_points, args.max_lag)
        if mpc_enabled
        else None
    )
    rows = []
    curve_records = []
    for index, (path, sample) in enumerate(zip(sample_paths, samples), start=1):
        connectivity = connectivity_metrics(sample, args.connectivity)
        direct = direct_similarity_metrics(training, sample)
        shifted = max_shifted_similarity(training, sample, args.max_shift)
        novelty8 = _patch_metrics_or_pending(
            training, sample, 8, args, args.seed + index
        )
        novelty16 = _patch_metrics_or_pending(
            training, sample, 16, args, args.seed + index
        )
        row = {
            "sample_index": index,
            "sample_path": str(path),
            "generated_porosity": float(sample.mean()),
            "connected_pore_fraction": connectivity["connected_pore_fraction"],
            "isolated_pore_fraction": connectivity["isolated_pore_fraction"],
            "percolating_fraction": connectivity["percolating_fraction"],
            "percolation_x": connectivity["percolation_x"],
            "percolation_y": connectivity["percolation_y"],
            "percolation_z": connectivity["percolation_z"],
            **direct,
            **shifted,
            "patch8_mean_nn_hamming": novelty8.get("mean_nearest_distance"),
            "patch8_exact_copy_rate": novelty8.get("exact_patch_match_rate"),
            "patch16_mean_nn_hamming": novelty16.get("mean_nearest_distance"),
            "patch16_exact_copy_rate": novelty16.get("exact_patch_match_rate"),
        }
        if args.target_porosity is not None:
            row["target_porosity"] = args.target_porosity
            row["absolute_condition_error"] = abs(sample.mean() - args.target_porosity)
            row["signed_condition_error"] = sample.mean() - args.target_porosity
        if mpc_enabled:
            sample_mpc = directional_mpc(sample, args.mpc_n_points, args.max_lag)
            dds = {
                axis: difference_degree(
                    reference_mpc["curves"][axis], sample_mpc["curves"][axis]
                )
                for axis in "xyz"
            }
            row.update(
                dd_x=dds["x"],
                dd_y=dds["y"],
                dd_z=dds["z"],
                dd_mean=float(np.mean(list(dds.values()))),
            )
            curve_records.append(
                {"sample_path": str(path), "curves": sample_mpc["curves"]}
            )
        else:
            row.update(dd_x="PENDING_AUTHOR_CONFIRMATION", dd_y="PENDING_AUTHOR_CONFIRMATION", dd_z="PENDING_AUTHOR_CONFIRMATION", dd_mean="PENDING_AUTHOR_CONFIRMATION")
        rows.append(row)

    metrics_path = output_dir / "metrics_per_sample.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric_metrics = [
        key
        for key, value in rows[0].items()
        if key not in {"sample_index"} and isinstance(value, (int, float, np.number))
    ]
    summary_rows = []
    for metric in numeric_metrics:
        values = [float(row[metric]) for row in rows]
        summary_rows.append({"metric": metric, **mean_sd_ci(values)})
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = {
        "sample_count": len(samples),
        "training_tiff": str(Path(args.training_tiff).resolve()),
        "training_porosity": float(training.mean()),
        "connectivity": args.connectivity,
        "mpc_status": (
            "manuscript_eq8_literal_v1"
            if mpc_enabled
            else "PENDING_AUTHOR_CONFIRMATION"
        ),
        "diversity": _diversity_summary(
            samples, mpc_enabled, args.mpc_n_points, args.max_lag
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mpc_enabled:
        (output_dir / "mpc_curves.json").write_text(
            json.dumps(
                {"reference": reference_mpc, "samples": curve_records},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(metrics_path)
    print(summary_path)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_tiff", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*_binary.tif")
    parser.add_argument("--target_porosity", type=float)
    parser.add_argument("--binary_threshold", type=float)
    parser.add_argument("--pore_is_high", type=_bool, default=True)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6)
    parser.add_argument("--mpc_definition", choices=["pending", "eq8_literal"], default="pending")
    parser.add_argument("--mpc_n_points", type=int, default=2)
    parser.add_argument("--max_lag", type=int)
    parser.add_argument("--max_shift", type=int, default=2)
    parser.add_argument("--training_patches", type=int, default=512)
    parser.add_argument("--generated_patches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _bool(value):
    return str(value).lower() in {"1", "true", "yes", "y"}


def _diversity_summary(samples, mpc_enabled, n_points, max_lag):
    if len(samples) < 2:
        return {"status": "NEED_AT_LEAST_TWO_SAMPLES"}
    if mpc_enabled:
        return pairwise_structural_diversity(samples, n_points, max_lag)
    summary = pairwise_hamming_diversity(samples)
    porosities = np.asarray([sample.mean() for sample in samples])
    differences = [
        abs(float(porosities[i] - porosities[j]))
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ]
    summary["mean_pairwise_porosity_difference"] = float(np.mean(differences))
    summary["mpc_component_status"] = "PENDING_AUTHOR_CONFIRMATION"
    return summary


def _patch_metrics_or_pending(training, sample, patch_size, args, seed):
    if min(training.shape) < patch_size:
        return {
            "status": "NOT_APPLICABLE_VOLUME_TOO_SMALL",
            "patch_size": patch_size,
        }
    return patch_novelty_metrics(
        training,
        sample,
        patch_size,
        args.training_patches,
        args.generated_patches,
        seed,
    )


if __name__ == "__main__":
    main()
