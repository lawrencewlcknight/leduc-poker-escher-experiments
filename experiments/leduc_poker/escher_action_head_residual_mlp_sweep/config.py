"""Configuration for residual trunks in the carried-forward ESCHER model.

This experiment revisits the residual-MLP question after the Experiment 22
regret-action-head improvement. The baseline is the carried-forward ESCHER
model: Experiment 13 training protocol, ``(256, 128)`` policy/regret/value
trunks, standard linear policy output, and one 64-unit separate per-action
head on the regret networks.
"""

from __future__ import annotations

from experiments.leduc_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config(
    "leduc_poker_escher_action_head_residual_mlp_sweep"
)
DEFAULT_CONFIG.update({
    "policy_network_head_depth": 0,
    "policy_network_head_units": None,
    "regret_network_head_depth": 1,
    "regret_network_head_units": 64,
})


def _variant(
    variant_id,
    variant_label,
    variant_description,
    layers,
    residual_mode,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": layers,
        "regret_network_layers": layers,
        "value_network_layers": layers,
        "policy_network_residual_mode": residual_mode,
        "regret_network_residual_mode": residual_mode,
        "value_network_residual_mode": residual_mode,
        "policy_network_head_depth": 0,
        "policy_network_head_units": None,
        "regret_network_head_depth": 1,
        "regret_network_head_units": 64,
    }


VARIANTS = [
    _variant(
        "carry_forward_256_128_action_heads",
        "Carry-forward 256x128",
        "Current carried-forward ESCHER architecture: 256x128 trunks and 64-unit regret action heads.",
        (256, 128),
        "same_width",
    ),
    _variant(
        "deep_plain_256_256_128_action_heads",
        "Deep plain 256x256x128",
        "Adds a repeated 256-unit trunk layer with residual connections disabled.",
        (256, 256, 128),
        "none",
    ),
    _variant(
        "deep_same_width_256_256_128_action_heads",
        "Deep same-width residual",
        "Matches the deep plain trunk but activates a same-width skip on the repeated 256-unit layer.",
        (256, 256, 128),
        "same_width",
    ),
    _variant(
        "bottleneck_plain_256_128_128_action_heads",
        "Bottleneck plain 256x128x128",
        "Adds a 128-unit bottleneck trunk layer with residual connections disabled.",
        (256, 128, 128),
        "none",
    ),
    _variant(
        "bottleneck_projection_256_128_128_action_heads",
        "Bottleneck projection residual",
        "Matches the bottleneck plain trunk but activates a projection skip on the width-changing 256-to-128 layer.",
        (256, 128, 128),
        "projection",
    ),
]

BASELINE_VARIANT_ID = "carry_forward_256_128_action_heads"
