"""Configuration checks for the ESCHER average-policy weighting ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_average_policy_weighting_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_average_policy_weighting_baseline_matches_carry_forward_config():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert DEFAULT_CONFIG["num_iterations"] == 80
    assert DEFAULT_CONFIG["num_traversals"] == 500
    assert DEFAULT_CONFIG["num_val_fn_traversals"] == 500
    assert DEFAULT_CONFIG["importance_sampling"] is False
    assert DEFAULT_CONFIG["zero_regret_fallback"] == "uniform"
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["regret_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["value_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["policy_network_head_depth"] == 0
    assert DEFAULT_CONFIG["regret_network_head_depth"] == 1
    assert DEFAULT_CONFIG["regret_network_head_units"] == 64
    assert DEFAULT_CONFIG["average_policy_weighting"] == "linear"
    assert DEFAULT_CONFIG["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_average_policy_weighting_variants_cover_deep_cfr_analogue():
    modes = {
        variant["variant_id"]: variant["average_policy_weighting"]
        for variant in VARIANTS
    }

    assert modes == {
        "linear_avg_weighting_baseline": "linear",
        "uniform_avg_weighting": "uniform",
    }


def test_average_policy_weighting_variants_are_configurable():
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)

        assert config["variant_id"] == variant["variant_id"]
        assert config["average_policy_weighting"] == variant["average_policy_weighting"]
        assert config["policy_network_head_depth"] == 0
        assert config["regret_network_head_depth"] == 1
        assert config["regret_network_head_units"] == 64
        assert config["total_policy_training_events_expected"] >= 1
