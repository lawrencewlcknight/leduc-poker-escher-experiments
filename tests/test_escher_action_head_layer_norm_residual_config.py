"""Configuration checks for the action-head LayerNorm/residual-LN ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_action_head_layer_norm_residual_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEED,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_action_head_layer_norm_residual_baseline_matches_carry_forward_config():
    assert DEFAULT_SEED == 1234
    assert DEFAULT_CONFIG["num_iterations"] == 80
    assert DEFAULT_CONFIG["num_traversals"] == 500
    assert DEFAULT_CONFIG["num_val_fn_traversals"] == 500
    assert DEFAULT_CONFIG["importance_sampling"] is False
    assert DEFAULT_CONFIG["zero_regret_fallback"] == "uniform"
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["regret_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["value_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["policy_network_head_depth"] == 0
    assert DEFAULT_CONFIG["policy_network_head_units"] is None
    assert DEFAULT_CONFIG["regret_network_head_depth"] == 1
    assert DEFAULT_CONFIG["regret_network_head_units"] == 64
    assert DEFAULT_CONFIG["regret_network_output_mode"] == "direct"
    assert DEFAULT_CONFIG["baseline_variant_id"] == BASELINE_VARIANT_ID


def test_action_head_layer_norm_residual_variants_keep_heads_fixed():
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)

        assert config["policy_network_head_depth"] == 0
        assert config["policy_network_head_units"] is None
        assert config["regret_network_head_depth"] == 1
        assert config["regret_network_head_units"] == 64
        assert config["regret_network_output_mode"] == "direct"


def test_action_head_layer_norm_residual_variants_cover_deep_cfr_analogue():
    by_id = {variant["variant_id"]: variant for variant in VARIANTS}

    baseline = by_id[BASELINE_VARIANT_ID]
    assert baseline["policy_network_layers"] == (256, 128)
    assert baseline["policy_network_layer_norm"] is True
    assert baseline["policy_network_residual_mode"] == "same_width"

    plain = by_id["plain_256_128_action_heads"]
    assert plain["policy_network_layers"] == (256, 128)
    assert plain["policy_network_layer_norm"] is False
    assert plain["policy_network_residual_mode"] == "none"

    deep_plain = by_id["deep_plain_256_256_128_action_heads"]
    deep_ln = by_id["deep_layer_norm_256_256_128_action_heads"]
    deep_residual_ln = by_id["deep_residual_layer_norm_256_256_128_action_heads"]

    assert deep_plain["policy_network_layers"] == (256, 256, 128)
    assert deep_plain["policy_network_layer_norm"] is False
    assert deep_plain["policy_network_residual_mode"] == "none"
    assert deep_ln["policy_network_layer_norm"] is True
    assert deep_ln["policy_network_residual_mode"] == "none"
    assert deep_residual_ln["policy_network_layer_norm"] is True
    assert deep_residual_ln["policy_network_residual_mode"] == "same_width"
