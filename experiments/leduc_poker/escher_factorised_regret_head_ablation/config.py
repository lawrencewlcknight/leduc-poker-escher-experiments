"""Configuration for factorised regret-output heads in ESCHER.

This experiment is the ESCHER analogue of the Deep CFR factorised advantage
head ablation. The baseline is the carried-forward ESCHER model: Experiment
13 training protocol, ``(256, 128)`` policy/regret/value trunks, standard
linear policy output, and one 64-unit separate per-action head on the regret
networks. Variants change only the regret-output factorisation.
"""

from __future__ import annotations

from experiments.leduc_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config(
    "leduc_poker_escher_factorised_regret_head_ablation"
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
    regret_network_output_mode,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_head_depth": 0,
        "policy_network_head_units": None,
        "regret_network_head_depth": 1,
        "regret_network_head_units": 64,
        "regret_network_output_mode": regret_network_output_mode,
    }


VARIANTS = [
    _variant(
        "direct_regret_action_head_64_baseline",
        "Direct regret action head",
        "Carry-forward ESCHER baseline: independent 64-unit per-action regret heads.",
        "direct",
    ),
    _variant(
        "centered_regret_action_head_64",
        "Centred regret action head",
        "Centres each regret head's legal-action outputs to zero mean per information state.",
        "centered",
    ),
    _variant(
        "dueling_regret_action_head_64",
        "Dueling regret action head",
        "Adds a scalar state-value head to centred legal-action regret deviations.",
        "dueling",
    ),
]

BASELINE_VARIANT_ID = "direct_regret_action_head_64_baseline"

DEFAULT_CONFIG.update({
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
})
