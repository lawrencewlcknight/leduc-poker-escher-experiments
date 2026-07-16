"""Configuration checks for Experiment 35 regret-target ablation."""

from __future__ import annotations

from escher_poker.regret_targets import AUTHOR_STATE_VALUE, PAPER_POLICY_WEIGHTED_Q
from experiments.leduc_poker.escher_regret_target_specification_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_regret_target_ablation_uses_experiment_28_candidate_baseline():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert DEFAULT_CONFIG["num_iterations"] == 80
    assert DEFAULT_CONFIG["num_traversals"] == 500
    assert DEFAULT_CONFIG["num_val_fn_traversals"] == 500
    assert DEFAULT_CONFIG["importance_sampling"] is False
    assert DEFAULT_CONFIG["zero_regret_fallback"] == "uniform"
    assert DEFAULT_CONFIG["all_actions"] is True
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 256, 128)
    assert DEFAULT_CONFIG["regret_network_layers"] == (256, 256, 128)
    assert DEFAULT_CONFIG["value_network_layers"] == (256, 256, 128)
    assert DEFAULT_CONFIG["regret_network_head_depth"] == 1
    assert DEFAULT_CONFIG["regret_network_head_units"] == 64
    assert DEFAULT_CONFIG["regret_target_processing"] == "standardize"
    assert DEFAULT_CONFIG["regret_target_baseline"] == AUTHOR_STATE_VALUE
    assert DEFAULT_CONFIG["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_regret_target_variants_cover_author_and_paper_definitions():
    settings = {
        variant["variant_id"]: variant["regret_target_baseline"]
        for variant in VARIANTS
    }
    assert settings == {
        "author_state_value_baseline": AUTHOR_STATE_VALUE,
        "paper_policy_weighted_q": PAPER_POLICY_WEIGHTED_Q,
    }


def test_regret_target_variants_change_only_target_baseline():
    baseline_config = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    ignored = {
        "variant_id",
        "variant_label",
        "variant_description",
        "regret_target_baseline",
    }

    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        for key, value in baseline_config.items():
            if key in ignored:
                continue
            assert config[key] == value
