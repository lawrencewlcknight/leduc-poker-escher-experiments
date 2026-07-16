"""Configuration checks for Experiment 41's combined candidate ablation."""

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
)
from experiments.leduc_poker.escher_combined_candidate_ablation.config import (
    BASELINE_VARIANT_ID,
    CANDIDATE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    MAXIMUM_STACK_VARIANT_ID,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def _configs_by_id():
    return {
        variant["variant_id"]: make_variant_config(DEFAULT_CONFIG, variant)
        for variant in VARIANTS
    }


def test_combined_candidate_uses_three_matched_seeds_and_three_arms():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert [variant["variant_id"] for variant in VARIANTS] == [
        BASELINE_VARIANT_ID,
        CANDIDATE_VARIANT_ID,
        MAXIMUM_STACK_VARIANT_ID,
    ]


def test_experiment_28_arm_preserves_the_baseline_algorithm():
    baseline = _configs_by_id()[BASELINE_VARIANT_ID]
    metadata = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "ablation_variants",
        "execution_backend",
    }
    for key, value in EXPERIMENT_28_CONFIG.items():
        if key not in metadata:
            assert baseline[key] == value


def test_candidate_and_maximum_stack_have_requested_settings():
    configs = _configs_by_id()
    candidate = configs[CANDIDATE_VARIANT_ID]
    maximum = configs[MAXIMUM_STACK_VARIANT_ID]

    for config in [candidate, maximum]:
        assert config["regret_target_baseline"] == "paper_policy_weighted_q"
        assert config["regret_target_processing"] == "standardize"
        assert config["regret_replay_mode"] == "infoset_stratified"
        assert config["execution_backend"] == "sequential"

    assert candidate["use_balanced_probs"] is False
    assert candidate["balanced_sampling_mix"] == 0.0
    assert maximum["use_balanced_probs"] is True
    assert maximum["balanced_sampling_mix"] == 1.0


def test_treatments_change_only_the_requested_algorithm_fields():
    configs = _configs_by_id()
    baseline = configs[BASELINE_VARIANT_ID]
    mutable = {
        "variant_id",
        "variant_label",
        "variant_description",
        "regret_target_baseline",
        "regret_replay_mode",
        "use_balanced_probs",
        "balanced_sampling_mix",
    }
    for variant_id in [CANDIDATE_VARIANT_ID, MAXIMUM_STACK_VARIANT_ID]:
        treatment = configs[variant_id]
        for key, value in baseline.items():
            if key not in mutable:
                assert treatment[key] == value
