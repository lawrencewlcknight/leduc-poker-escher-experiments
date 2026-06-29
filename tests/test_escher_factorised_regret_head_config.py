"""Configuration checks for the ESCHER factorised regret-head ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_factorised_regret_head_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEED,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_factorised_regret_head_baseline_matches_carry_forward_config():
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


def test_factorised_regret_head_variants_cover_deep_cfr_analogue():
    modes = {
        variant["variant_id"]: variant["regret_network_output_mode"]
        for variant in VARIANTS
    }

    assert modes == {
        "direct_regret_action_head_64_baseline": "direct",
        "centered_regret_action_head_64": "centered",
        "dueling_regret_action_head_64": "dueling",
    }


def test_factorised_regret_head_variants_keep_other_architecture_fixed():
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)

        assert config["variant_id"] == variant["variant_id"]
        assert config["policy_network_head_depth"] == 0
        assert config["policy_network_head_units"] is None
        assert config["regret_network_head_depth"] == 1
        assert config["regret_network_head_units"] == 64
        assert config["regret_network_output_mode"] == variant["regret_network_output_mode"]
        assert config["policy_network_layers"] == (256, 128)
        assert config["regret_network_layers"] == (256, 128)
        assert config["value_network_layers"] == (256, 128)
