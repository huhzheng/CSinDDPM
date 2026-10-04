import numpy as np
import pytest

from evaluation.condition_control import calibration_summary
from evaluation.connectivity import connectivity_metrics
from evaluation.dd import difference_degree
from evaluation.diversity import pairwise_hamming_diversity
from evaluation.mpc import equation8_mpc_curve
from evaluation.novelty import (
    direct_similarity_metrics,
    max_shifted_similarity,
    patch_novelty_metrics,
)


def test_directional_connectivity_uses_explicit_axis_order():
    volume = np.zeros((8, 8, 8), dtype=bool)
    volume[3, 4, :] = True
    metrics = connectivity_metrics(volume, connectivity=6)
    assert metrics["axis_order"] == "zyx"
    assert metrics["directional_spanning"]["x"]["spanning_component_count"] == 1
    assert metrics["directional_spanning"]["y"]["spanning_component_count"] == 0
    assert metrics["directional_spanning"]["z"]["spanning_component_count"] == 0


def test_literal_eq8_and_eq9_are_numerically_defined():
    ones = np.ones((5, 5, 5), dtype=bool)
    curve = equation8_mpc_curve(ones, axis=2, n_points=3, max_lag=2)
    assert np.allclose(curve, 1.0)
    assert difference_degree(curve, np.array([1.0, 0.5, 0.0])) == pytest.approx(1.25)


def test_novelty_and_diversity_use_complete_binary_semantics():
    zeros = np.zeros((8, 8, 8), dtype=bool)
    ones = np.ones((8, 8, 8), dtype=bool)
    novelty = patch_novelty_metrics(ones, ones, patch_size=4, seed=4)
    assert novelty["exact_patch_match_rate"] == 1.0
    diversity = pairwise_hamming_diversity([zeros, ones])
    assert diversity["mean_pairwise_hamming"] == 1.0
    direct = direct_similarity_metrics(zeros, ones)
    assert direct["normalized_hamming_distance"] == 1.0


def test_shift_robust_similarity_recovers_periodic_translation():
    training = np.zeros((8, 8, 8), dtype=bool)
    training[1:3, 2:5, 4:7] = True
    generated = np.roll(training, shift=(1, -2, 2), axis=(0, 1, 2))
    result = max_shifted_similarity(training, generated, max_shift=2)
    assert result["max_shifted_similarity"] == 1.0


def test_condition_calibration_reports_local_and_global_behavior():
    summary = calibration_summary(
        [0.17, 0.17, 0.18, 0.18, 0.35, 0.35],
        [0.171, 0.169, 0.181, 0.179, 0.22, 0.23],
        training_porosity=0.18,
        local_radius=0.03,
    )
    groups = {row["target"]: row for row in summary["by_target"]}
    assert groups[0.18]["within_local_training_range"]
    assert not groups[0.35]["within_local_training_range"]
    assert summary["mae"] > 0
