"""Configuration checks for Experiment 39 fixed sampling coverage."""

from __future__ import annotations

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    TEMPERED_BALANCED_MIX,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_sampling_ablation_baseline_preserves_experiment_28_algorithm():
    baseline = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    experimental_keys = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "ablation_variants",
        "track_sampling_coverage",
    }
    for key, value in CANDIDATE_DEFAULT_CONFIG.items():
        if key not in experimental_keys:
            assert baseline[key] == value
    assert baseline["variant_id"] == BASELINE_VARIANT_ID
    assert baseline["use_balanced_probs"] is False
    assert baseline["balanced_sampling_mix"] == 0.0
    assert baseline["track_sampling_coverage"] is True


def test_sampling_ablation_covers_requested_fixed_policies():
    settings = {
        variant["variant_id"]: (
            variant["use_balanced_probs"],
            variant["balanced_sampling_mix"],
        )
        for variant in VARIANTS
    }
    assert settings == {
        BASELINE_VARIANT_ID: (False, 0.0),
        "exact_balanced_fixed_sampling": (True, 1.0),
        "tempered_balanced_fixed_sampling": (True, TEMPERED_BALANCED_MIX),
    }


def test_sampling_arms_hold_all_other_experiment_28_settings_fixed():
    baseline = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    varied = {
        "variant_id",
        "variant_label",
        "variant_description",
        "use_balanced_probs",
        "balanced_sampling_mix",
    }
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        for key, value in baseline.items():
            if key not in varied:
                assert config[key] == value

        assert config["expl"] == 1.0
        assert config["importance_sampling"] is False
        assert config["track_sampling_coverage"] is True

    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
