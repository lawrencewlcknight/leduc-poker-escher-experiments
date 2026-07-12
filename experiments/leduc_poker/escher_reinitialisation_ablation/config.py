"""Configuration for the Experiment 28 ESCHER reinitialisation ablation.

The baseline is exactly the Experiment 28 candidate architecture. The treatment
keeps all architecture, target-processing, traversal, replay, and optimisation
settings fixed while disabling regret-network and value-network
reinitialisation between ESCHER iterations.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)

DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "candidate_reinitialised"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    reinitialize_regret_networks,
    reinitialize_value_network,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "reinitialize_regret_networks": reinitialize_regret_networks,
        "reinitialize_value_network": reinitialize_value_network,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Experiment 28 reinitialised baseline",
        (
            "Exact Experiment 28 candidate architecture with regret and value "
            "networks reinitialised according to the carried-forward protocol."
        ),
        True,
        True,
    ),
    _variant(
        "candidate_no_reinitialisation",
        "No regret/value reinitialisation",
        (
            "Experiment 28 candidate architecture with regret and value "
            "networks kept persistent across ESCHER iterations."
        ),
        False,
        False,
    ),
]

DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_reinitialisation_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": "Experiment 28 reinitialised baseline",
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "reinitialize_regret_networks": True,
    "reinitialize_value_network": True,
})
