"""Configuration checks for Experiment 42's 10x-node candidate ablation."""

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
)
from experiments.leduc_poker.escher_long_horizon_candidate_ablation.config import (
    BASELINE_VARIANT_ID,
    CANDIDATE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_28_SOLVE_PASSES,
    LONG_RUN_NUM_ITERATIONS,
    LONG_RUN_SOLVE_PASSES,
    SOLVE_PASS_MULTIPLIER,
    TARGET_SOLVE_PASS_MULTIPLIER,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def _configs_by_id():
    return {
        variant["variant_id"]: make_variant_config(DEFAULT_CONFIG, variant)
        for variant in VARIANTS
    }


def test_long_horizon_experiment_has_two_arms_and_three_seeds():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert [variant["variant_id"] for variant in VARIANTS] == [
        BASELINE_VARIANT_ID,
        CANDIDATE_VARIANT_ID,
    ]


def test_long_horizon_is_approximately_ten_times_experiment_28_solve_passes():
    assert EXPERIMENT_28_SOLVE_PASSES == 81
    assert TARGET_SOLVE_PASS_MULTIPLIER == 10
    assert LONG_RUN_SOLVE_PASSES == 801
    assert LONG_RUN_NUM_ITERATIONS == 800
    assert SOLVE_PASS_MULTIPLIER == 801 / 81
    assert DEFAULT_CONFIG["num_iterations"] == 800


def test_long_run_baseline_changes_only_training_horizon_and_metadata():
    baseline = _configs_by_id()[BASELINE_VARIANT_ID]
    allowed = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "ablation_variants",
        "execution_backend",
        "num_iterations",
        "intermediate_policy_training_events_expected",
        "total_policy_training_events_expected",
        "policy_gradient_steps_expected",
        "target_solve_pass_multiplier",
        "solve_pass_multiplier",
        "experiment_28_solve_passes",
        "long_run_solve_passes",
    }
    for key, value in EXPERIMENT_28_CONFIG.items():
        if key not in allowed:
            assert baseline[key] == value


def test_long_run_candidate_has_requested_settings_only():
    configs = _configs_by_id()
    baseline = configs[BASELINE_VARIANT_ID]
    candidate = configs[CANDIDATE_VARIANT_ID]
    assert candidate["regret_target_baseline"] == "paper_policy_weighted_q"
    assert candidate["regret_target_processing"] == "standardize"
    assert candidate["regret_replay_mode"] == "infoset_stratified"
    assert candidate["use_balanced_probs"] is False
    assert candidate["balanced_sampling_mix"] == 0.0
    assert candidate["execution_backend"] == "sequential"

    changed = {
        key
        for key in baseline
        if baseline.get(key) != candidate.get(key)
    }
    assert changed == {
        "variant_id",
        "variant_label",
        "variant_description",
        "regret_target_baseline",
        "regret_replay_mode",
    }
