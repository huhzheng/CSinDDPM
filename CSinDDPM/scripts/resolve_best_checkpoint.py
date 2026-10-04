"""Resolve and verify a checkpoint selected by the fixed validation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(
    train_dir: Path,
    variant: str,
    *,
    require_complete: bool = True,
) -> tuple[Path, dict]:
    train_dir = train_dir.expanduser().resolve()
    manifest_path = train_dir / "best_checkpoints.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"selection manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_complete and manifest.get("status") != "complete":
        raise RuntimeError(
            f"selection manifest is not complete: status={manifest.get('status')!r}"
        )

    if variant == "overall":
        selected = manifest.get("best_overall")
    else:
        selected = manifest.get("best_by_variant", {}).get(variant)
    if not selected:
        available = ["overall", *manifest.get("best_by_variant", {}).keys()]
        raise KeyError(
            f"variant {variant!r} is unavailable; choose one of {sorted(available)}"
        )

    checkpoint = train_dir / selected["alias"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint alias is missing: {checkpoint}")
    expected_hash = selected.get("alias_sha256")
    actual_hash = sha256_file(checkpoint)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "selected checkpoint hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}, path={checkpoint}"
        )
    return checkpoint, selected


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve best_model, best EMA, or best-overall weights from a "
            "completed best_checkpoints.json manifest."
        )
    )
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument(
        "--variant",
        default="model",
        help="model, overall, or an EMA key such as ema_0.9999",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow a running manifest (intended only for diagnostics)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the selected record as JSON instead of only its path",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        help="write a sampling-side provenance record for this selection",
    )
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    checkpoint, selected = resolve_checkpoint(
        args.train_dir,
        args.variant,
        require_complete=not args.allow_incomplete,
    )
    payload = dict(selected)
    payload.update(
        requested_variant=args.variant,
        resolved_path=str(checkpoint),
        selection_manifest=str(
            args.train_dir.expanduser().resolve() / "best_checkpoints.json"
        ),
    )
    if args.record_output:
        output = args.record_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(checkpoint)


if __name__ == "__main__":
    main()
