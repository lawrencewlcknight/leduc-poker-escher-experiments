"""Configuration checks for the ESCHER action-head residual-MLP sweep."""

from __future__ import annotations

from experiments.leduc_poker.escher_action_head_residual_mlp_sweep.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEED,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_action_head_residual_baseline_matches_carry_forward_config():
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
    assert BASELINE_VARIANT_ID == "carry_forward_256_128_action_heads"


def test_action_head_residual_variants_keep_regret_heads_fixed():
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)

        assert config["policy_network_head_depth"] == 0
        assert config["policy_network_head_units"] is None
        assert config["regret_network_head_depth"] == 1
        assert config["regret_network_head_units"] == 64


def test_action_head_residual_variants_cover_plain_and_residual_controls():
    by_id = {variant["variant_id"]: variant for variant in VARIANTS}

    assert by_id[BASELINE_VARIANT_ID]["policy_network_layers"] == (256, 128)
    assert by_id["deep_plain_256_256_128_action_heads"]["policy_network_residual_mode"] == "none"
    assert by_id["deep_same_width_256_256_128_action_heads"]["policy_network_residual_mode"] == "same_width"
    assert by_id["bottleneck_plain_256_128_128_action_heads"]["policy_network_residual_mode"] == "none"
    assert by_id["bottleneck_projection_256_128_128_action_heads"]["policy_network_residual_mode"] == "projection"
    assert (
        by_id["bottleneck_plain_256_128_128_action_heads"]["policy_network_layers"]
        == by_id["bottleneck_projection_256_128_128_action_heads"]["policy_network_layers"]
    )
