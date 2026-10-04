"""Preflight and verify the three-data best-checkpoint rerun artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "sandstone": ROOT / "data/sandstone_64_uint8/min_0_0_0.tif",
    "carbonate": ROOT / "data/CarbonateRock/carbonate_min_448_256_384.tif",
    "mt_gambier_limestone": (
        ROOT / "data/Mt Gambier limestone/mt_gambier_z256_y384_x128.tif"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_data(errors: list[str]) -> dict:
    records = {}
    hashes = []
    for name, path in DATASETS.items():
        record = {"path": str(path)}
        if not path.is_file():
            errors.append(f"missing data TIFF: {path}")
            records[name] = record
            continue
        array = tifffile.imread(path)
        unique = np.unique(array)
        digest = sha256_file(path)
        hashes.append(digest)
        record.update(
            shape=list(array.shape),
            dtype=str(array.dtype),
            unique_values=[float(value) for value in unique],
            porosity_high_phase=float((array > 0).mean()),
            sha256=digest,
            byte_size=path.stat().st_size,
        )
        if array.shape != (64, 64, 64):
            errors.append(f"{name}: expected shape (64,64,64), got {array.shape}")
        if not np.issubdtype(array.dtype, np.integer):
            errors.append(f"{name}: expected an integer TIFF, got {array.dtype}")
        if set(unique.tolist()) != {0, 255}:
            errors.append(
                f"{name}: expected exact binary values {{0,255}}, got {unique.tolist()}"
            )
        records[name] = record
    if len(hashes) != len(set(hashes)):
        errors.append("the three training TIFF files do not have three distinct hashes")
    return records


def expected_steps(train_steps: int, save_interval: int) -> list[int]:
    steps = list(range(save_interval, train_steps + 1, save_interval))
    if not steps or steps[-1] != train_steps:
        steps.append(train_steps)
    return steps


def _parse_csv_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def audit_disabled_dirty_log(
    run_dir: Path,
    key: str,
    errors: list[str],
) -> dict:
    """Prove that a zero-probability dirty augmentation was never applied."""

    initial_error_count = len(errors)
    log_path = run_dir / "train_log.csv"
    audit = {
        "path": str(log_path),
        "expected_dirty_applied_count": 0,
        "expected_dirty_mode": "none",
        "expected_cuboid_bbox": "null",
        "expected_porosity_change_count": 0,
    }
    if not log_path.is_file():
        errors.append(f"{key}: missing train_log.csv for clean-data verification")
        audit["status"] = "FAIL"
        return audit

    with log_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {
        "step",
        "dirty_applied",
        "dirty_mode",
        "cuboid_bbox",
        "porosity_before",
        "porosity_after",
    }
    missing = sorted(required - fieldnames)
    audit["row_count"] = len(rows)
    audit["missing_fields"] = missing
    if missing:
        errors.append(f"{key}: train_log.csv missing clean-audit fields {missing}")
        audit["status"] = "FAIL"
        return audit
    if not rows:
        errors.append(f"{key}: train_log.csv is empty")
        audit["status"] = "FAIL"
        return audit

    dirty_applied_count = 0
    invalid_boolean_count = 0
    non_none_mode_count = 0
    non_null_bbox_count = 0
    porosity_change_count = 0
    malformed_porosity_count = 0
    steps = []
    invalid_step_count = 0
    for row in rows:
        try:
            steps.append(int(row["step"]))
        except (TypeError, ValueError):
            invalid_step_count += 1
        try:
            dirty_applied_count += int(_parse_csv_bool(row["dirty_applied"]))
        except ValueError:
            invalid_boolean_count += 1
        if str(row["dirty_mode"]).strip().lower() != "none":
            non_none_mode_count += 1
        if str(row["cuboid_bbox"]).strip().lower() not in {"null", "none", ""}:
            non_null_bbox_count += 1
        try:
            before = float(row["porosity_before"])
            after = float(row["porosity_after"])
        except (TypeError, ValueError):
            malformed_porosity_count += 1
        else:
            if not np.isclose(before, after, rtol=0.0, atol=1e-12):
                porosity_change_count += 1

    audit.update(
        dirty_applied_count=dirty_applied_count,
        invalid_boolean_count=invalid_boolean_count,
        non_none_mode_count=non_none_mode_count,
        non_null_bbox_count=non_null_bbox_count,
        porosity_change_count=porosity_change_count,
        malformed_porosity_count=malformed_porosity_count,
        invalid_step_count=invalid_step_count,
        unique_step_count=len(set(steps)),
        duplicate_step_row_count=len(steps) - len(set(steps)),
        step_min=min(steps) if steps else None,
        step_max=max(steps) if steps else None,
    )
    violations = {
        "dirty_applied": dirty_applied_count,
        "invalid_dirty_applied": invalid_boolean_count,
        "dirty_mode_not_none": non_none_mode_count,
        "cuboid_bbox_not_null": non_null_bbox_count,
        "porosity_changed": porosity_change_count,
        "malformed_porosity": malformed_porosity_count,
        "invalid_step": invalid_step_count,
    }
    if any(violations.values()):
        errors.append(f"{key}: clean-data train_log.csv violations {violations}")
    audit["status"] = "PASS" if len(errors) == initial_error_count else "FAIL"
    return audit


def verify_protocol_metadata(
    run_dir: Path,
    key: str,
    args,
    errors: list[str],
    *,
    include_dirty: bool,
) -> dict:
    """Verify declared model/augmentation settings when expectations are supplied."""

    expectations_requested = any(
        value is not None
        for value in (
            args.expected_experiment_variant,
            args.expected_use_condition,
            args.expected_dirty_probability if include_dirty else None,
            args.expected_dirty_min_frac if include_dirty else None,
            args.expected_dirty_max_frac if include_dirty else None,
        )
    )
    if not expectations_requested:
        return {"status": "NOT_REQUESTED"}

    initial_error_count = len(errors)
    audit = {"status": "CHECKED"}
    config_path = run_dir / "config.json"
    summary_path = run_dir / "model_summary.json"
    if not config_path.is_file():
        errors.append(f"{key}: missing config.json for protocol verification")
        return {"status": "FAIL", "config_path": str(config_path)}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit["config_path"] = str(config_path)

    if args.expected_experiment_variant is not None:
        actual = config.get("experiment_variant")
        audit["experiment_variant"] = actual
        if actual != args.expected_experiment_variant:
            errors.append(
                f"{key}: experiment_variant is {actual!r}, expected "
                f"{args.expected_experiment_variant!r}"
            )

    expected_condition = (
        args.expected_use_condition == "true"
        if args.expected_use_condition is not None
        else None
    )
    if expected_condition is not None:
        actual = config.get("use_condition")
        audit["use_condition"] = actual
        if actual is not expected_condition:
            errors.append(
                f"{key}: use_condition is {actual!r}, expected "
                f"{expected_condition!r}"
            )
        if not expected_condition and config.get("target_porosity") is not None:
            errors.append(
                f"{key}: unconditional sampling config retained a target_porosity"
            )

    if include_dirty:
        dirty_expectations = {
            "dirty_probability": args.expected_dirty_probability,
            "dirty_min_frac": args.expected_dirty_min_frac,
            "dirty_max_frac": args.expected_dirty_max_frac,
        }
        for field, expected in dirty_expectations.items():
            if expected is None:
                continue
            actual = config.get(field)
            audit[field] = actual
            if actual is None or not np.isclose(float(actual), expected):
                errors.append(
                    f"{key}: {field} is {actual!r}, expected {expected!r}"
                )
        expected_probability = getattr(args, "expected_dirty_probability", None)
        if expected_probability is not None and np.isclose(
            expected_probability, 0.0
        ):
            audit["clean_training_log"] = audit_disabled_dirty_log(
                run_dir, key, errors
            )

    if expected_condition is None:
        audit["status"] = (
            "PASS" if len(errors) == initial_error_count else "FAIL"
        )
        return audit
    if not summary_path.is_file():
        errors.append(f"{key}: missing model_summary.json for protocol verification")
        audit["status"] = "FAIL"
        return audit

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit["model_summary_path"] = str(summary_path)
    fields = (
        "use_condition",
        "condition_encoder_parameter_count",
        "conditional_resblock_count",
        "conditional_resblock_parameter_count",
        "conditional_parameter_count",
        "condition_fusion",
    )
    audit["model_summary"] = {field: summary.get(field) for field in fields}
    if summary.get("use_condition") is not expected_condition:
        errors.append(f"{key}: model summary condition flag does not match protocol")
    encoder_count = summary.get("condition_encoder_parameter_count")
    block_count = summary.get("conditional_resblock_count")
    block_parameter_count = summary.get("conditional_resblock_parameter_count")
    conditional_parameter_count = summary.get("conditional_parameter_count")
    if expected_condition:
        if not isinstance(encoder_count, int) or encoder_count <= 0:
            errors.append(f"{key}: conditioned model has no condition encoder parameters")
        if not isinstance(block_count, int) or block_count <= 0:
            errors.append(f"{key}: conditioned model has no conditional ResBlocks")
        if not isinstance(block_parameter_count, int) or block_parameter_count <= 0:
            errors.append(f"{key}: conditioned model has no ResBlock condition parameters")
        if not isinstance(conditional_parameter_count, int) or conditional_parameter_count <= 0:
            errors.append(f"{key}: conditioned model has zero conditional parameters")
        if summary.get("condition_fusion") == "disabled":
            errors.append(f"{key}: conditioned model reports disabled fusion")
    else:
        expected_zero = {
            "condition_encoder_parameter_count": encoder_count,
            "conditional_resblock_count": block_count,
            "conditional_resblock_parameter_count": block_parameter_count,
            "conditional_parameter_count": conditional_parameter_count,
        }
        for field, actual in expected_zero.items():
            if actual != 0:
                errors.append(f"{key}: {field} is {actual!r}, expected 0")
        if summary.get("condition_fusion") != "disabled":
            errors.append(f"{key}: unconditional model does not report disabled fusion")
    audit["status"] = "PASS" if len(errors) == initial_error_count else "FAIL"
    return audit


def verify_training(args, data_records: dict, errors: list[str]) -> dict:
    root = args.train_root.expanduser().resolve()
    expected = expected_steps(args.train_steps, args.save_interval)
    expected_variants = {"model", f"ema_{args.ema_rate}"}
    records = {}
    for name in DATASETS:
        for seed in args.seeds:
            key = f"{name}/train_seed_{seed}"
            train_dir = root / name / f"train_seed_{seed}"
            record = {"train_dir": str(train_dir)}
            record["protocol_audit"] = verify_protocol_metadata(
                train_dir, key, args, errors, include_dirty=True
            )
            clean_log_audit = record["protocol_audit"].get(
                "clean_training_log"
            )
            if clean_log_audit is not None:
                complete_step_coverage = (
                    clean_log_audit.get("row_count", 0) >= args.train_steps
                    and clean_log_audit.get("unique_step_count") == args.train_steps
                    and clean_log_audit.get("step_min") == 1
                    and clean_log_audit.get("step_max") == args.train_steps
                    and clean_log_audit.get("invalid_step_count") == 0
                )
                clean_log_audit["minimum_row_count"] = args.train_steps
                clean_log_audit["expected_unique_step_count"] = args.train_steps
                clean_log_audit["expected_step_range"] = [1, args.train_steps]
                clean_log_audit["complete_step_coverage"] = complete_step_coverage
                if not complete_step_coverage:
                    errors.append(
                        f"{key}: clean train_log.csv does not cover every step "
                        f"from 1 through {args.train_steps}"
                    )
                    clean_log_audit["status"] = "FAIL"
                    record["protocol_audit"]["status"] = "FAIL"
            manifest_path = train_dir / "best_checkpoints.json"
            final_model = train_dir / f"model{args.train_steps:06d}.pt"
            if not manifest_path.is_file():
                errors.append(f"{key}: missing {manifest_path.name}")
                records[key] = record
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["manifest_status"] = manifest.get("status")
            record["final_model_exists"] = final_model.is_file()
            if manifest.get("status") != "complete":
                errors.append(
                    f"{key}: selection manifest status is {manifest.get('status')!r}"
                )
            if not final_model.is_file():
                errors.append(f"{key}: missing final model {final_model.name}")

            protocol = manifest.get("selection_protocol", {})
            if protocol.get("metric") != f"mean_{args.metric}":
                errors.append(
                    f"{key}: selection metric is {protocol.get('metric')!r}, "
                    f"expected mean_{args.metric}"
                )
            if int(protocol.get("validation_trials", -1)) != args.validation_trials:
                errors.append(f"{key}: validation trial count does not match")
            if int(protocol.get("validation_seed", -1)) != args.validation_seed:
                errors.append(f"{key}: validation seed does not match")
            if protocol.get("validation_source_role") != (
                "training_volume_fixed_noise_proxy"
            ):
                errors.append(f"{key}: unexpected validation source role")
            data_hash = data_records.get(name, {}).get("sha256")
            if data_hash and protocol.get("validation_source_sha256") != data_hash:
                errors.append(f"{key}: validation TIFF hash differs from preflight")

            candidates = manifest.get("candidates", [])
            actual_keys = {
                (int(candidate["step"]), candidate["variant"])
                for candidate in candidates
            }
            expected_keys = {
                (step, variant)
                for step in expected
                for variant in expected_variants
            }
            missing_keys = sorted(expected_keys - actual_keys)
            duplicate_count = len(candidates) - len(actual_keys)
            if missing_keys:
                errors.append(f"{key}: missing validation candidates {missing_keys}")
            if duplicate_count:
                errors.append(f"{key}: duplicate validation candidates found")

            selections = dict(manifest.get("best_by_variant", {}))
            selections["overall"] = manifest.get("best_overall")
            selected_records = {}
            for variant, selected in selections.items():
                if not selected:
                    errors.append(f"{key}: missing best selection for {variant}")
                    continue
                alias = train_dir / selected["alias"]
                source = train_dir / selected["source_checkpoint"]
                if not alias.is_file() or not source.is_file():
                    errors.append(
                        f"{key}: selected files missing for {variant}: "
                        f"alias={alias.is_file()}, source={source.is_file()}"
                    )
                    continue
                actual_hash = sha256_file(alias)
                if actual_hash != selected.get("alias_sha256"):
                    errors.append(f"{key}: alias hash mismatch for {variant}")
                if actual_hash != sha256_file(source):
                    errors.append(f"{key}: alias/source mismatch for {variant}")
                selected_records[variant] = {
                    "step": selected["step"],
                    "variant": selected["variant"],
                    "score": selected["score"],
                    "alias": str(alias),
                    "source": str(source),
                    "sha256": actual_hash,
                }
            record.update(
                expected_candidate_count=len(expected_keys),
                actual_candidate_count=len(candidates),
                selected=selected_records,
            )
            records[key] = record
    return records


def _duplicate_hash_groups(paths: list[Path], hashes: list[str]) -> list[dict]:
    groups: dict[str, list[str]] = {}
    for path, digest in zip(paths, hashes):
        groups.setdefault(digest, []).append(path.name)
    return [
        {"sha256": digest, "sample_files": sample_files}
        for digest, sample_files in sorted(groups.items())
        if len(sample_files) > 1
    ]


def _record_integrity_issue(
    message: str,
    policy: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if policy == "warn":
        warnings.append(message)
    else:
        errors.append(message)


def verify_sampling(
    args,
    errors: list[str],
    warnings: list[str] | None = None,
) -> dict:
    if warnings is None:
        warnings = []
    root = args.sample_root.expanduser().resolve()
    binary_duplicate_policy = getattr(
        args, "binary_duplicate_policy", "error"
    )
    exact_training_copy_policy = getattr(
        args, "exact_training_copy_policy", "error"
    )
    records = {}
    for name in DATASETS:
        training_binary = (
            tifffile.imread(DATASETS[name])
            if DATASETS[name].is_file()
            else None
        )
        for seed in args.seeds:
            key = f"{name}/train_seed_{seed}/{args.variant}"
            sample_dir = root / name / f"train_seed_{seed}" / args.variant
            record = {"sample_dir": str(sample_dir)}
            record["protocol_audit"] = verify_protocol_metadata(
                sample_dir, key, args, errors, include_dirty=False
            )
            binary_paths = sorted(sample_dir.glob("sample_*_binary.tif"))
            raw_paths = sorted(sample_dir.glob("sample_*_raw.tif"))
            if len(binary_paths) != args.samples_per_dataset:
                errors.append(
                    f"{key}: expected {args.samples_per_dataset} binary TIFFs, "
                    f"found {len(binary_paths)}"
                )
            if len(raw_paths) != args.samples_per_dataset:
                errors.append(
                    f"{key}: expected {args.samples_per_dataset} raw TIFFs, "
                    f"found {len(raw_paths)}"
                )
            binary_hashes = []
            raw_hashes = []
            exact_training_copy_files = []
            for path in binary_paths:
                array = tifffile.imread(path)
                if array.shape != (64, 64, 64):
                    errors.append(f"{key}: invalid binary shape in {path.name}")
                if set(np.unique(array).tolist()).difference({0, 255}):
                    errors.append(f"{key}: non-binary values in {path.name}")
                binary_hashes.append(sha256_file(path))
                if training_binary is not None and np.array_equal(
                    array, training_binary
                ):
                    exact_training_copy_files.append(path.name)
            for path in raw_paths:
                array = tifffile.imread(path)
                if array.shape != (64, 64, 64) or not np.isfinite(array).all():
                    errors.append(f"{key}: invalid raw volume in {path.name}")
                raw_hashes.append(sha256_file(path))
            duplicate_binary_groups = _duplicate_hash_groups(
                binary_paths, binary_hashes
            )
            if duplicate_binary_groups:
                _record_integrity_issue(
                    f"{key}: duplicate binary sample hashes found: "
                    f"{duplicate_binary_groups}",
                    binary_duplicate_policy,
                    errors,
                    warnings,
                )
            if exact_training_copy_files:
                _record_integrity_issue(
                    f"{key}: exact voxel copies of the training TIFF found: "
                    f"{exact_training_copy_files}",
                    exact_training_copy_policy,
                    errors,
                    warnings,
                )
            if raw_hashes and len(set(raw_hashes)) != len(raw_hashes):
                errors.append(f"{key}: duplicate raw sample hashes found")

            log_path = sample_dir / "sampling_log.csv"
            rows = []
            if log_path.is_file():
                with log_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != args.samples_per_dataset:
                    errors.append(f"{key}: sampling_log.csv row count mismatch")
                seeds = [int(row["seed"]) for row in rows]
                expected_seeds = list(
                    range(
                        args.sampling_seed,
                        args.sampling_seed + args.samples_per_dataset,
                    )
                )
                if seeds != expected_seeds:
                    errors.append(f"{key}: sampling seeds do not match protocol")
            else:
                errors.append(f"{key}: missing sampling_log.csv")

            selection_path = sample_dir / "checkpoint_selection.json"
            if not selection_path.is_file():
                errors.append(f"{key}: missing checkpoint_selection.json")
            else:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                if selection.get("requested_variant") != args.variant:
                    errors.append(f"{key}: copied selection variant mismatch")
                resolved_checkpoint = Path(selection.get("resolved_path", ""))
                if not resolved_checkpoint.is_file():
                    errors.append(f"{key}: resolved checkpoint no longer exists")
                elif sha256_file(resolved_checkpoint) != selection.get("alias_sha256"):
                    errors.append(f"{key}: resolved checkpoint hash mismatch")

                config_path = sample_dir / "config.json"
                if not config_path.is_file():
                    errors.append(f"{key}: missing sampling config.json")
                else:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    configured_checkpoint = Path(
                        config.get("resolved_model_path", "")
                    )
                    if configured_checkpoint != resolved_checkpoint:
                        errors.append(f"{key}: sampled checkpoint differs from selection")
                    if int(config.get("seed", -1)) != args.sampling_seed:
                        errors.append(f"{key}: config sampling seed mismatch")
                    if int(config.get("num_samples", -1)) != args.samples_per_dataset:
                        errors.append(f"{key}: config sample count mismatch")
            record.update(
                binary_count=len(binary_paths),
                raw_count=len(raw_paths),
                unique_binary_hashes=len(set(binary_hashes)),
                unique_raw_hashes=len(set(raw_hashes)),
                duplicate_binary_groups=duplicate_binary_groups,
                exact_training_copy_count=len(exact_training_copy_files),
                exact_training_copy_files=exact_training_copy_files,
                binary_duplicate_policy=binary_duplicate_policy,
                exact_training_copy_policy=exact_training_copy_policy,
                sampling_log_rows=len(rows),
            )
            records[key] = record
    return records


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["data", "training", "sampling", "all"], default="all"
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=ROOT / "revision_outputs/full_model_baseline_3data",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=ROOT / "revision_outputs/full_model_samples_seed1001_1010_3data",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--train-steps", type=int, default=60000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--ema-rate", default="0.9999")
    parser.add_argument("--metric", choices=["mse", "vb", "total_loss"], default="total_loss")
    parser.add_argument("--validation-trials", type=int, default=16)
    parser.add_argument("--validation-seed", type=int, default=314159)
    parser.add_argument("--variant", default="model")
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=1001)
    parser.add_argument(
        "--binary-duplicate-policy",
        choices=["error", "warn"],
        default="error",
        help=(
            "Keep the default strict failure, or record duplicates as warnings "
            "when evaluating every frozen sample without replacement."
        ),
    )
    parser.add_argument(
        "--exact-training-copy-policy",
        choices=["error", "warn"],
        default="error",
        help=(
            "Keep the default strict failure, or retain exact training-volume "
            "copies as documented warnings for all-sample evaluation."
        ),
    )
    parser.add_argument("--expected-experiment-variant")
    parser.add_argument(
        "--expected-use-condition", choices=["true", "false"]
    )
    parser.add_argument("--expected-dirty-probability", type=float)
    parser.add_argument("--expected-dirty-min-frac", type=float)
    parser.add_argument("--expected-dirty-max-frac", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    data_records = inspect_data(errors)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "scope": (
            "Engineering/data-integrity verification only; PASS is not evidence "
            "of morphology fidelity, transport fidelity, or manuscript readiness."
        ),
        "data": data_records,
    }
    if args.stage in {"training", "all"}:
        report["training"] = verify_training(args, data_records, errors)
    if args.stage in {"sampling", "all"}:
        report["sampling"] = verify_sampling(args, errors, warnings)
    report["errors"] = errors
    report["warnings"] = warnings
    report["status"] = (
        "FAIL"
        if errors
        else "PASS_WITH_WARNINGS"
        if warnings
        else "PASS"
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"{report['status']}: {output}")
    else:
        print(rendered, end="")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
