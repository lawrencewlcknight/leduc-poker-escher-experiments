"""Configuration checks for Experiment 30 learning-rate ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_learning_rate_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_learning_rate_ablation_uses_experiment_28_candidate_baseline():
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


def test_learning_rate_variants_change_only_constant_learning_rate():
    learning_rates = {
        variant["variant_id"]: variant["learning_rate"]
        for variant in VARIANTS
    }
    assert learning_rates == {
        "candidate_lr_1e_3": 1e-3,
        "candidate_lr_5e_4": 5e-4,
        "candidate_lr_2e_3": 2e-3,
    }

    baseline_config = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        assert config["learning_rate_schedule"] == "constant"
        assert config["learning_rate_end"] == config["learning_rate"]
        assert config["learning_rate_decay_rate"] == 1.0
        assert config["learning_rate_warmup_iterations"] == 0

        ignored = {
            "variant_id",
            "variant_label",
            "variant_description",
            "learning_rate",
            "learning_rate_end",
        }
        for key, value in baseline_config.items():
            if key in ignored:
                continue
            assert config[key] == value
