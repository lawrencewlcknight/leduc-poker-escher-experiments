"""Configuration checks for Experiment 37's 2x2 correction factorial."""

from __future__ import annotations

from escher_poker.regret_target_processing import BATCH_RMS, BATCH_STANDARDIZE
from escher_poker.regret_targets import AUTHOR_STATE_VALUE, PAPER_POLICY_WEIGHTED_Q
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_regret_target_factorial_correction.config import (
    BASELINE_VARIANT_ID,
    BOTH_VARIANT_ID,
    CONFIRMATION_SEEDS,
    DEFAULT_CONFIG,
    MEANINGFUL_SUCCESS_THRESHOLD,
    POLICY_ONLY_VARIANT_ID,
    SCALE_ONLY_VARIANT_ID,
    SCREENING_SEEDS,
    TARGET_NODES,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_factorial_baseline_exactly_preserves_experiment_28_training_config():
    baseline = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    metadata_keys = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "factorial_variants",
        "policy_weighted_q_correction",
        "scale_only_normalization",
        "target_nodes",
        "meaningful_success_threshold",
        "confirmation_top_k",
    }
    for key, value in CANDIDATE_DEFAULT_CONFIG.items():
        if key not in metadata_keys:
            assert baseline[key] == value
    assert baseline["regret_target_baseline"] == AUTHOR_STATE_VALUE
    assert baseline["regret_target_processing"] == BATCH_STANDARDIZE
    assert baseline["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_factorial_has_all_four_target_by_normalization_cells():
    settings = {
        variant["variant_id"]: (
            variant["regret_target_baseline"],
            variant["regret_target_processing"],
        )
        for variant in VARIANTS
    }
    assert settings == {
        BASELINE_VARIANT_ID: (AUTHOR_STATE_VALUE, BATCH_STANDARDIZE),
        POLICY_ONLY_VARIANT_ID: (PAPER_POLICY_WEIGHTED_Q, BATCH_STANDARDIZE),
        SCALE_ONLY_VARIANT_ID: (AUTHOR_STATE_VALUE, BATCH_RMS),
        BOTH_VARIANT_ID: (PAPER_POLICY_WEIGHTED_Q, BATCH_RMS),
    }


def test_factorial_uses_independent_three_then_five_seed_stages():
    assert len(SCREENING_SEEDS) == 3
    assert len(CONFIRMATION_SEEDS) >= 5
    assert set(SCREENING_SEEDS).isdisjoint(CONFIRMATION_SEEDS)
    assert TARGET_NODES == 1_000_000.0
    assert MEANINGFUL_SUCCESS_THRESHOLD == 0.3
