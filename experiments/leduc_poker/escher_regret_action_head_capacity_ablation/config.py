"""Configuration for the Experiment 28 regret action-head capacity ablation.

The baseline is exactly the Experiment 28 candidate architecture. Treatments
hold the trunk, policy output, target processing, replay settings, and training
protocol fixed while increasing only the regret action-head capacity.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)

DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "baseline_regret_head_64"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    regret_network_head_depth,
    regret_network_head_units,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_head_depth": 0,
        "policy_network_head_units": None,
        "regret_network_head_depth": regret_network_head_depth,
        "regret_network_head_units": regret_network_head_units,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Regret head 64",
        "Exact Experiment 28 candidate architecture with one 64-unit per-action regret head.",
        1,
        64,
    ),
    _variant(
        "regret_head_128",
        "Regret head 128",
        "Widens the per-action regret head from 64 to 128 units.",
        1,
        128,
    ),
    _variant(
        "regret_head_64_64",
        "Regret head 64-64",
        "Deepens the per-action regret head to two 64-unit layers.",
        2,
        64,
    ),
]

DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_action_head_capacity_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": "Regret head 64",
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
})
