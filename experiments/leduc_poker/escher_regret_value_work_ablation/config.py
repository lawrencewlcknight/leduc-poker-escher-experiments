"""Configuration for the Experiment 28 ESCHER regret/value work ablation.

The baseline is exactly the Experiment 28 candidate architecture. Treatments
hold architecture, replay, target processing, and learning rate fixed while
shifting data collection or supervised fitting effort from value learning
toward regret learning.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)

DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "baseline_regret_value_work"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    num_traversals,
    num_val_fn_traversals,
    regret_network_train_steps,
    value_network_train_steps,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "num_traversals": num_traversals,
        "num_val_fn_traversals": num_val_fn_traversals,
        "regret_network_train_steps": regret_network_train_steps,
        "value_network_train_steps": value_network_train_steps,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Baseline 500/500, 200/200",
        "Exact Experiment 28 candidate architecture and work allocation.",
        500,
        500,
        200,
        200,
    ),
    _variant(
        "regret_data_heavy",
        "Regret-data heavy",
        (
            "Keeps the nominal traversal budget approximately matched while "
            "shifting collection from value traversals to regret traversals."
        ),
        625,
        250,
        200,
        200,
    ),
    _variant(
        "regret_update_heavy",
        "Regret-update heavy",
        (
            "Keeps the combined regret/value supervised update budget fixed "
            "while shifting fitting effort toward the regret networks."
        ),
        500,
        500,
        300,
        100,
    ),
    _variant(
        "regret_data_and_update_heavy",
        "Regret-data + update heavy",
        (
            "Combines the regret-heavy traversal allocation with the "
            "regret-heavy supervised update allocation."
        ),
        625,
        250,
        300,
        100,
    ),
]

DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_value_work_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": "Baseline 500/500, 200/200",
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
})
