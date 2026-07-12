"""Configuration checks for Experiment 29 reinitialisation ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_reinitialisation_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_reinitialisation_ablation_uses_experiment_28_candidate_baseline():
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
    assert DEFAULT_CONFIG["policy_network_layer_norm"] is False
    assert DEFAULT_CONFIG["regret_network_layer_norm"] is False
    assert DEFAULT_CONFIG["value_network_layer_norm"] is False
    assert DEFAULT_CONFIG["policy_network_residual_mode"] == "none"
    assert DEFAULT_CONFIG["regret_network_residual_mode"] == "none"
    assert DEFAULT_CONFIG["value_network_residual_mode"] == "none"
    assert DEFAULT_CONFIG["policy_network_head_depth"] == 0
    assert DEFAULT_CONFIG["policy_network_head_units"] is None
    assert DEFAULT_CONFIG["regret_network_head_depth"] == 1
    assert DEFAULT_CONFIG["regret_network_head_units"] == 64
    assert DEFAULT_CONFIG["regret_network_output_mode"] == "direct"
    assert DEFAULT_CONFIG["regret_target_processing"] == "standardize"
    assert DEFAULT_CONFIG["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_reinitialisation_ablation_changes_only_reinitialisation_flags():
    baseline, treatment = VARIANTS
    baseline_config = make_variant_config(DEFAULT_CONFIG, baseline)
    treatment_config = make_variant_config(DEFAULT_CONFIG, treatment)

    assert baseline_config["reinitialize_regret_networks"] is True
    assert baseline_config["reinitialize_value_network"] is True
    assert treatment_config["reinitialize_regret_networks"] is False
    assert treatment_config["reinitialize_value_network"] is False

    ignored = {
        "variant_id",
        "variant_label",
        "variant_description",
        "reinitialize_regret_networks",
        "reinitialize_value_network",
    }
    for key, value in baseline_config.items():
        if key in ignored:
            continue
        assert treatment_config[key] == value
