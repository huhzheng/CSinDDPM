"""Run a tiny real train/checkpoint/sample cycle inside revision_outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.resolve_best_checkpoint import resolve_checkpoint


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "revision_outputs/cli_integration_smoke" / run_id
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "tiny_input.tif"
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[:, :, :4] = 255
    tifffile.imwrite(data_path, volume)
    train_dir = output / "train"
    common = [
        "--data_dir",
        str(data_path),
        "--image_size",
        "8",
        "--channel_mult",
        "1",
        "--attention_resolutions",
        "8",
        "--num_channels",
        "32",
        "--num_head_channels",
        "32",
        "--num_res_blocks",
        "1",
        "--diffusion_steps",
        "10",
        "--noise_schedule",
        "cosine",
        "--learn_sigma",
        "false",
        "--seed",
        "123",
    ]
    train_command = [
        sys.executable,
        str(ROOT / "demo_train.py"),
        *common,
        "--output_dir",
        str(train_dir),
        "--train_steps",
        "2",
        "--save_interval",
        "1",
        "--log_interval",
        "1",
        "--dirty_probability",
        "1",
        "--select_best_checkpoint",
        "true",
        "--validation_trials",
        "2",
        "--validation_seed",
        "77",
        "--min_size",
        "8",
        "--max_size",
        "8",
    ]
    subprocess.run(train_command, cwd=ROOT, check=True)
    checkpoint = train_dir / "best_model.pt"
    manifest_path = train_dir / "best_checkpoints.json"
    expected_training_outputs = [
        checkpoint,
        train_dir / "best_ema_0.9999.pt",
        train_dir / "best_overall.pt",
        train_dir / "validation_log.csv",
        manifest_path,
        train_dir / "compute_train.json",
    ]
    if not all(path.is_file() for path in expected_training_outputs):
        raise AssertionError("tiny training did not produce checkpoint/compute metadata")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_keys = {
        (candidate["step"], candidate["variant"])
        for candidate in manifest["candidates"]
    }
    if manifest["status"] != "complete" or candidate_keys != {
        (1, "model"),
        (1, "ema_0.9999"),
        (2, "model"),
        (2, "ema_0.9999"),
    }:
        raise AssertionError("tiny training did not complete best-checkpoint selection")
    resolved, selection = resolve_checkpoint(train_dir, "model")
    if resolved != checkpoint.resolve():
        raise AssertionError("best-checkpoint resolver selected an unexpected file")
    train_log = pd.read_csv(train_dir / "train_log.csv")
    if list(train_log["step"]) != [1, 2]:
        raise AssertionError("tiny training log is not exactly two steps")
    if not train_log["dirty_applied"].all():
        raise AssertionError("forced dirty augmentation did not execute")
    model_summary = json.loads(
        (train_dir / "model_summary.json").read_text(encoding="utf-8")
    )
    if not model_summary["use_condition"] or not (
        model_summary["conditional_resblock_count"]
        == model_summary["resblock_count"] > 0
    ):
        raise AssertionError("tiny training did not use the complete conditional model")

    sample_dir = output / "samples"
    sample_command = [
        sys.executable,
        str(ROOT / "demo_sample_num.py"),
        *common,
        "--model_path",
        str(checkpoint),
        "--output_dir",
        str(sample_dir),
        "--num_samples",
        "1",
        "--target_porosity",
        "0.5",
        "--progress",
        "false",
    ]
    subprocess.run(sample_command, cwd=ROOT, check=True)
    expected = [
        sample_dir / "sample_0001_binary.tif",
        sample_dir / "sample_0001_raw.tif",
        sample_dir / "sampling_log.csv",
        sample_dir / "compute_sample.json",
    ]
    if not all(path.is_file() for path in expected):
        raise AssertionError("tiny sampling did not produce all audited outputs")
    sampling_log = pd.read_csv(sample_dir / "sampling_log.csv")
    generated_porosity = float(sampling_log.loc[0, "generated_porosity"])
    if not 0.0 <= generated_porosity <= 1.0:
        raise AssertionError("generated porosity is outside [0, 1]")

    evaluation_dir = output / "evaluation"
    evaluation_command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_samples.py"),
        "--training_tiff",
        str(data_path),
        "--sample_dir",
        str(sample_dir),
        "--output_dir",
        str(evaluation_dir),
        "--target_porosity",
        "0.5",
        "--max_shift",
        "1",
        "--training_patches",
        "32",
        "--generated_patches",
        "16",
    ]
    subprocess.run(evaluation_command, cwd=ROOT, check=True)
    if not (evaluation_dir / "metrics_per_sample.csv").is_file():
        raise AssertionError("tiny evaluation did not produce per-sample metrics")

    report = {
        "scope": "tiny local functional integration; not generation-quality evidence",
        "checkpoint_selection": selection,
        "compute_train": json.loads(
            (train_dir / "compute_train.json").read_text(encoding="utf-8")
        ),
        "train_command": train_command,
        "sample_command": sample_command,
        "evaluation_command": evaluation_command,
        "checkpoint": str(checkpoint),
        "generated_porosity": generated_porosity,
        "status": "PASS",
    }
    report_path = output / "integration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {report_path}")


if __name__ == "__main__":
    main()
