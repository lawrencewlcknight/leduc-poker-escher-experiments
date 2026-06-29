"""Configuration for the ESCHER average-policy weighting ablation.

This experiment tests the Deep CFR Experiment 10 average-strategy weighting
idea in ESCHER. The baseline is the carried-forward ESCHER configuration: the
Experiment 13 training protocol, ``(256, 128)`` policy/regret/value trunks, no
importance-sampling correction, uniform zero-regret fallback, and one 64-unit
separate per-action head on the regret networks. Variants differ only in the
sample weighting used when training the supervised average-policy network.
"""

from __future__ import annotations

from experiments.leduc_poker.escher_architecture_base import make_default_config

DEFAULT_SEEDS = [1234, 2025, 31415]
DEFAULT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]


def _variant(
    variant_id,
    variant_label,
    variant_description,
    average_policy_weighting,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "average_policy_weighting": average_policy_weighting,
    }


VARIANTS = [
    _variant(
        "linear_avg_weighting_baseline",
        "Linear average-policy weighting",
        "Carry-forward ESCHER baseline: average-policy loss weighted by CFR iteration.",
        "linear",
    ),
    _variant(
        "uniform_avg_weighting",
        "Uniform average-policy weighting",
        "Removes CFR iteration weighting from the average-policy supervised loss.",
        "uniform",
    ),
]

BASELINE_VARIANT_ID = "linear_avg_weighting_baseline"

DEFAULT_CONFIG = make_default_config(
    "leduc_poker_escher_average_policy_weighting_ablation"
)
DEFAULT_CONFIG.update({
    "policy_network_head_depth": 0,
    "policy_network_head_units": None,
    "regret_network_head_depth": 1,
    "regret_network_head_units": 64,
    "average_policy_weighting": "linear",
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
})
