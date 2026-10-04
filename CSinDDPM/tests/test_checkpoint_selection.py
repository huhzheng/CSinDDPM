import hashlib
import json
from pathlib import Path
import uuid

import pytest

from demo_diffusion.demo_train_util import TrainLoop
from scripts.resolve_best_checkpoint import resolve_checkpoint


def test_candidate_order_prefers_score_then_earlier_step_then_model():
    current = {"score": 1.0, "step": 10, "variant": "ema_0.9999"}
    assert TrainLoop._candidate_is_better(
        {"score": 0.9, "step": 20, "variant": "ema_0.9999"}, current
    )
    assert TrainLoop._candidate_is_better(
        {"score": 1.0, "step": 9, "variant": "ema_0.9999"}, current
    )
    assert TrainLoop._candidate_is_better(
        {"score": 1.0, "step": 10, "variant": "model"}, current
    )


def test_resolver_requires_complete_manifest_and_matching_hash():
    workspace_tmp = Path(__file__).resolve().parents[1] / "revision_outputs"
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    tmp_path = workspace_tmp / f"checkpoint-selection-test-{uuid.uuid4().hex}"
    tmp_path.mkdir()
    try:
        checkpoint = tmp_path / "best_model.pt"
        checkpoint.write_bytes(b"checkpoint")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        selected = {
            "alias": checkpoint.name,
            "alias_sha256": digest,
            "score": 0.5,
            "step": 5000,
            "variant": "model",
        }
        manifest = {
            "status": "complete",
            "best_by_variant": {"model": selected},
            "best_overall": selected,
        }
        (tmp_path / "best_checkpoints.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        resolved, record = resolve_checkpoint(tmp_path, "model")
        assert resolved == checkpoint.resolve()
        assert record["step"] == 5000

        checkpoint.write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="hash mismatch"):
            resolve_checkpoint(tmp_path, "model")
    finally:
        for path in tmp_path.iterdir():
            path.unlink()
        tmp_path.rmdir()
