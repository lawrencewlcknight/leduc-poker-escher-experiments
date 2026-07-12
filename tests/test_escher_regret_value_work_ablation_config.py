"""Configuration checks for Experiment 32 regret/value work ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_regret_value_work_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_regret_value_work_ablation_uses_experiment_28_candidate_baseline():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert DEFAULT_CONFIG["num_iterations"] == 80
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
    assert DEFAULT_CONFIG["memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["batch_size_regret"] == 256
    assert DEFAULT_CONFIG["batch_size_value"] == 256
    assert DEFAULT_CONFIG["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_regret_value_work_variants_match_requested_rebalance_schedule():
    settings = {
        variant["variant_id"]: (
            variant["num_traversals"],
            variant["num_val_fn_traversals"],
            variant["regret_network_train_steps"],
            variant["value_network_train_steps"],
        )
        for variant in VARIANTS
    }

    assert settings == {
        "baseline_regret_value_work": (500, 500, 200, 200),
        "regret_data_heavy": (625, 250, 200, 200),
        "regret_update_heavy": (500, 500, 300, 100),
        "regret_data_and_update_heavy": (625, 250, 300, 100),
    }


def test_regret_value_work_variants_change_only_work_allocation():
    baseline_config = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    ignored = {
        "variant_id",
        "variant_label",
        "variant_description",
        "num_traversals",
        "num_val_fn_traversals",
        "regret_network_train_steps",
        "value_network_train_steps",
    }

    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        for key, value in baseline_config.items():
            if key in ignored:
                continue
            assert config[key] == value
