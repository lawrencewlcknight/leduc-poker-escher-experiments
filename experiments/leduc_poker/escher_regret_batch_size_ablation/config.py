"""Configuration for the Experiment 28 ESCHER regret-batch-size ablation.

The baseline is exactly the Experiment 28 candidate architecture. The treatment
keeps replay capacity at 50k and changes only the regret-network supervised
batch size from 256 to 512.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)

DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "baseline_regret_batch_256"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    batch_size_regret,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "memory_capacity": 50_000,
        "batch_size_regret": batch_size_regret,
        "batch_size_value": 256,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Baseline regret batch 256",
        "Exact Experiment 28 candidate architecture with 50k replay and regret/value batch 256.",
        256,
    ),
    _variant(
        "regret_batch_512",
        "Regret batch 512",
        "Increases regret-network batch size to 512 while keeping replay capacity at 50k and value batch at 256.",
        512,
    ),
]

DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_batch_size_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": "Baseline regret batch 256",
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "memory_capacity": 50_000,
    "batch_size_regret": 256,
    "batch_size_value": 256,
})
