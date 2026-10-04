"""Regression checks for the public full-model, non-ablation distribution."""

import json
from pathlib import Path

import pytest
import torch

from demo_diffusion.demo_sinddpm import ResBlock, UNetModel
from demo_diffusion.reproducibility import build_model_summary
from demo_sample_num import create_argparser as sample_parser
from demo_train import create_argparser as train_parser
from scripts.verify_best_3data import create_argparser as verify_parser


ROOT = Path(__file__).resolve().parents[1]


def test_full_model_manifest_and_cli_defaults_agree():
    protocol = json.loads(
        (ROOT / "configs/best_checkpoint_rerun_3data.json").read_text(encoding="utf-8")
    )
    train = train_parser().parse_args([])
    for field, expected in protocol["training"].items():
        # The three-data wrapper sets microbatch=1; the generic CLI uses -1 (all).
        if field == "microbatch":
            continue
        actual = getattr(train, field)
        if field == "ema_rate":
            actual = float(actual)
        assert actual == expected, field
    assert all(protocol["module_state"].values())
    assert train.select_best_checkpoint is True
    assert sample_parser().parse_args([]).seed == protocol["sampling"]["seed_start"]
    assert verify_parser().parse_args([]).sampling_seed == 1001
    assert protocol["checkpoint_selection"]["default_reconstruction_variant"] == "model"


def test_model_contains_all_conditional_branches():
    model = UNetModel(
        image_size=8, in_channels=1, model_channels=32, out_channels=2,
        num_res_blocks=1, attention_resolutions=(), channel_mult=(1,),
    )
    summary = build_model_summary(model)
    assert summary["use_condition"] is True
    assert summary["condition_encoder_parameter_count"] > 0
    assert summary["conditional_resblock_count"] == summary["resblock_count"] > 0
    assert summary["conditional_resblock_parameter_count"] > 0
    assert summary["condition_fusion"] == "additive_residual_fusion"


def test_full_model_resblock_rejects_disabled_condition():
    with pytest.raises(ValueError, match="requires use_condition=true"):
        ResBlock(32, 128, 0.0, use_condition=False)


def test_porosity_changes_output_when_conditional_weights_are_active():
    torch.manual_seed(2026)
    model = UNetModel(
        image_size=8, in_channels=1, model_channels=32, out_channels=2,
        num_res_blocks=1, attention_resolutions=(), channel_mult=(1,),
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("out_layers.3.weight") or name == "out.2.weight":
                parameter.normal_(0, 0.01)
        noise = torch.randn((1, 1, 8, 8, 8))
        timestep = torch.tensor([123])
        low = model(noise, timestep, labels=torch.tensor([[0.10]]))
        high = model(noise, timestep, labels=torch.tensor([[0.45]]))
    assert torch.isfinite(low).all() and torch.isfinite(high).all()
    assert not torch.equal(low, high)


def test_three_data_wrappers_enforce_full_model_and_best_checkpoint():
    train = (ROOT / "scripts/train_best_3data.sh").read_text(encoding="utf-8")
    sample = (ROOT / "scripts/sample_best_3data.sh").read_text(encoding="utf-8")
    for script in (train, sample):
        assert 'USE_CONDITION="${USE_CONDITION:-true}"' in script
        assert 'if [[ "$USE_CONDITION" != "true" ]]' in script
        assert "STRICT_PROTOCOL=1" in script
        assert 'DIRTY_PROBABILITY="${DIRTY_PROBABILITY:-0.30}"' in script
    assert "--select_best_checkpoint true" in train
    assert "--target_porosity" in sample
    assert "SAMPLING_SEED:-1001" in sample
    assert "CHECKPOINT_VARIANT:-model" in sample
    assert "resolve_best_checkpoint.py" in sample
    assert not list((ROOT / "scripts").glob("*ablation*"))
    assert not list((ROOT / "configs").glob("*ablation*"))


def test_integrity_warnings_are_not_silently_enabled():
    args = verify_parser().parse_args([])
    assert args.binary_duplicate_policy == "error"
    assert args.exact_training_copy_policy == "error"
