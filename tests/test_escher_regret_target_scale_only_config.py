"""Configuration checks for Experiment 36 target-scaling ablation."""

from __future__ import annotations

from escher_poker.regret_target_processing import (
    BATCH_RMS,
    BATCH_STANDARDIZE,
    EMA_STD,
    FIXED_UTILITY_SCALE,
    RAW,
)
from escher_poker.regret_targets import (
    AUTHOR_STATE_VALUE,
    PAPER_POLICY_WEIGHTED_Q,
)
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_regret_target_scale_only_ablation.config import (
    BASELINE_VARIANT_ID,
    CORRECTED_STANDARDIZED_CONTROL_ID,
    DEFAULT_CONFIG,
    DEFAULT_EMA_DECAY,
    DEFAULT_SEEDS,
    LEDUC_UTILITY_RANGE,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_scale_only_ablation_uses_exact_experiment_28_training_configuration():
    baseline = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    metadata_keys = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "ablation_variants",
        "regret_target_fixed_scale",
        "regret_target_ema_decay",
    }

    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    for key, value in CANDIDATE_DEFAULT_CONFIG.items():
        if key not in metadata_keys:
            assert baseline[key] == value
    assert baseline["regret_target_baseline"] == AUTHOR_STATE_VALUE
    assert baseline["regret_target_processing"] == BATCH_STANDARDIZE
    assert baseline["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_corrected_processing_arms_cover_requested_mean_free_scales():
    settings = {
        variant["variant_id"]: (
            variant["regret_target_baseline"],
            variant["regret_target_processing"],
        )
        for variant in VARIANTS
    }

    assert settings == {
        BASELINE_VARIANT_ID: (AUTHOR_STATE_VALUE, BATCH_STANDARDIZE),
        CORRECTED_STANDARDIZED_CONTROL_ID: (
            PAPER_POLICY_WEIGHTED_Q,
            BATCH_STANDARDIZE,
        ),
        "corrected_raw": (PAPER_POLICY_WEIGHTED_Q, RAW),
        "corrected_fixed_utility_scale": (
            PAPER_POLICY_WEIGHTED_Q,
            FIXED_UTILITY_SCALE,
        ),
        "corrected_batch_rms": (PAPER_POLICY_WEIGHTED_Q, BATCH_RMS),
        "corrected_persistent_std": (PAPER_POLICY_WEIGHTED_Q, EMA_STD),
    }


def test_corrected_arms_change_only_target_and_processing_choices():
    control = make_variant_config(DEFAULT_CONFIG, VARIANTS[1])
    ignored = {
        "variant_id",
        "variant_label",
        "variant_description",
        "regret_target_processing",
    }

    for variant in VARIANTS[1:]:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        for key, value in control.items():
            if key not in ignored:
                assert config[key] == value

    assert LEDUC_UTILITY_RANGE == 26.0
    assert DEFAULT_EMA_DECAY == 0.99
