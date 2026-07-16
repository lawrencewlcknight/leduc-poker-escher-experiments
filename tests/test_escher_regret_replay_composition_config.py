"""Configuration checks for Experiment 38 regret replay ablation."""

from __future__ import annotations

from escher_poker.replay import (
    ALL_SAMPLES,
    COUNTERFACTUAL_REACH_WEIGHTED,
    INFOSET_STRATIFIED,
    RARE_HISTORY_QUOTA,
    RESERVOIR,
)
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_regret_replay_composition_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_replay_ablation_baseline_exactly_preserves_experiment_28():
    baseline = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    metadata_keys = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "ablation_variants",
    }
    for key, value in CANDIDATE_DEFAULT_CONFIG.items():
        if key not in metadata_keys:
            assert baseline[key] == value
    assert baseline["regret_replay_mode"] == RESERVOIR
    assert baseline["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_replay_ablation_covers_requested_composition_strategies():
    settings = {
        variant["variant_id"]: variant["regret_replay_mode"]
        for variant in VARIANTS
    }
    assert settings == {
        BASELINE_VARIANT_ID: RESERVOIR,
        "all_regret_samples": ALL_SAMPLES,
        "infoset_stratified_replay": INFOSET_STRATIFIED,
        "rare_history_quota_replay": RARE_HISTORY_QUOTA,
        "counterfactual_reach_weighted_replay": COUNTERFACTUAL_REACH_WEIGHTED,
    }


def test_replay_arms_hold_optimizer_work_and_all_other_settings_fixed():
    baseline = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    ignored = {
        "variant_id",
        "variant_label",
        "variant_description",
        "regret_replay_mode",
    }
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        for key, value in baseline.items():
            if key not in ignored:
                assert config[key] == value

    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert DEFAULT_CONFIG["memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["batch_size_regret"] == 256
    assert DEFAULT_CONFIG["regret_network_train_steps"] == 200
