"""Configuration for the Experiment 28 ESCHER replay-capacity ablation.

The baseline is exactly the Experiment 28 candidate architecture. Treatments
hold architecture, target processing, traversal budget, and supervised update
counts fixed while varying replay capacity and regret batch size.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)

DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "baseline_replay_50k"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    memory_capacity,
    batch_size_regret,
    batch_size_value,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "memory_capacity": memory_capacity,
        "batch_size_regret": batch_size_regret,
        "batch_size_value": batch_size_value,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Baseline replay 50k",
        "Exact Experiment 28 candidate architecture with 50k replay and regret/value batch 256.",
        50_000,
        256,
        256,
    ),
    _variant(
        "medium_replay_100k",
        "Medium replay 100k",
        "Doubles replay capacity while leaving regret and value batch sizes unchanged.",
        100_000,
        256,
        256,
    ),
    _variant(
        "large_replay_200k",
        "Large replay 200k",
        "Quadruples replay capacity while leaving regret and value batch sizes unchanged.",
        200_000,
        256,
        256,
    ),
    _variant(
        "large_replay_200k_regret_batch_512",
        "Large replay 200k + regret batch 512",
        "Combines 200k replay capacity with a larger regret batch while holding value batch at 256.",
        200_000,
        512,
        256,
    ),
]

DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_replay_capacity_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": "Baseline replay 50k",
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "memory_capacity": 50_000,
    "batch_size_regret": 256,
    "batch_size_value": 256,
})
