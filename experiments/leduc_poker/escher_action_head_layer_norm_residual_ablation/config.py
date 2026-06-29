"""Configuration for layer-normalised and residual-LN action-head ESCHER.

This experiment is the ESCHER analogue of the Deep CFR layer-normalisation
network ablation, but uses the carried-forward ESCHER action-head baseline.
The baseline is the Experiment 13 training protocol, ``(256, 128)``
policy/regret/value trunks, standard linear policy output, and one 64-unit
separate per-action head on the regret networks. Variants isolate plain,
layer-normalised, and residual-plus-layer-normalised trunk parameterisations.
"""

from __future__ import annotations

from experiments.leduc_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config(
    "leduc_poker_escher_action_head_layer_norm_residual_ablation"
)
DEFAULT_CONFIG.update({
    "policy_network_head_depth": 0,
    "policy_network_head_units": None,
    "regret_network_head_depth": 1,
    "regret_network_head_units": 64,
    "regret_network_output_mode": "direct",
})


def _variant(
    variant_id,
    variant_label,
    variant_description,
    layers,
    layer_norm,
    residual_mode,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": layers,
        "regret_network_layers": layers,
        "value_network_layers": layers,
        "policy_network_layer_norm": bool(layer_norm),
        "regret_network_layer_norm": bool(layer_norm),
        "value_network_layer_norm": bool(layer_norm),
        "policy_network_residual_mode": residual_mode,
        "regret_network_residual_mode": residual_mode,
        "value_network_residual_mode": residual_mode,
        "policy_network_head_depth": 0,
        "policy_network_head_units": None,
        "regret_network_head_depth": 1,
        "regret_network_head_units": 64,
        "regret_network_output_mode": "direct",
    }


VARIANTS = [
    _variant(
        "carry_forward_layer_norm_256_128_action_heads",
        "Carry-forward LayerNorm 256x128",
        "Current carried-forward ESCHER model: layer-normalised 256x128 trunks and 64-unit regret action heads.",
        (256, 128),
        True,
        "same_width",
    ),
    _variant(
        "plain_256_128_action_heads",
        "Plain 256x128",
        "Removes layer normalisation from the carried-forward trunk without changing capacity.",
        (256, 128),
        False,
        "none",
    ),
    _variant(
        "deep_plain_256_256_128_action_heads",
        "Deep plain 256x256x128",
        "Adds a repeated 256-unit trunk layer with layer normalisation and residual connections disabled.",
        (256, 256, 128),
        False,
        "none",
    ),
    _variant(
        "deep_layer_norm_256_256_128_action_heads",
        "Deep LayerNorm 256x256x128",
        "Adds the same repeated 256-unit trunk layer while retaining layer normalisation.",
        (256, 256, 128),
        True,
        "none",
    ),
    _variant(
        "deep_residual_layer_norm_256_256_128_action_heads",
        "Deep residual+LayerNorm 256x256x128",
        "Matches the deep LayerNorm trunk but activates a same-width residual connection on the repeated 256-unit layer.",
        (256, 256, 128),
        True,
        "same_width",
    ),
]

BASELINE_VARIANT_ID = "carry_forward_layer_norm_256_128_action_heads"

DEFAULT_CONFIG.update({
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
})
