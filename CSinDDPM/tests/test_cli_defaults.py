from demo_sample_num import create_argparser as sample_parser
from demo_train import create_argparser as train_parser


def test_training_defaults_match_submitted_method_claims():
    args = train_parser().parse_args([])
    assert args.experiment_variant == "A_full"
    assert args.train_steps == 60000
    assert args.dirty_probability == 0.30
    assert args.dirty_min_frac == 0.10
    assert args.dirty_max_frac == 0.30
    assert args.learn_sigma is True
    assert args.use_condition is True
    assert args.select_best_checkpoint is True
    assert args.validation_trials == 16
    assert args.validation_seed == 314159
    assert args.best_checkpoint_metric == "total_loss"
    assert args.validate_ema is True


def test_sampling_requires_explicit_checkpoint_at_runtime():
    args = sample_parser().parse_args([])
    assert args.experiment_variant == "A_full"
    assert args.model_path == ""
    assert args.target_porosity == 0.18
    assert args.seed == 1001
