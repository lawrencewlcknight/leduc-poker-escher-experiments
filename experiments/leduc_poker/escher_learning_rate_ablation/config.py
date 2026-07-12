"""Configuration for the Experiment 28 ESCHER learning-rate ablation.

The baseline is exactly the Experiment 28 candidate architecture. Treatments
hold architecture, reinitialisation, target processing, traversal budget,
replay, and supervised update budgets fixed while changing only the constant
learning rate.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)

DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "candidate_lr_1e_3"


def _variant(variant_id, variant_label, variant_description, learning_rate):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "learning_rate": learning_rate,
        "learning_rate_schedule": "constant",
        "learning_rate_end": learning_rate,
        "learning_rate_decay_rate": 1.0,
        "learning_rate_warmup_iterations": 0,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Baseline LR 1e-3",
        "Exact Experiment 28 candidate architecture with constant learning rate 1e-3.",
        1e-3,
    ),
    _variant(
        "candidate_lr_5e_4",
        "Low LR 5e-4",
        "Experiment 28 candidate architecture with constant learning rate 5e-4.",
        5e-4,
    ),
    _variant(
        "candidate_lr_2e_3",
        "High LR 2e-3",
        "Experiment 28 candidate architecture with constant learning rate 2e-3.",
        2e-3,
    ),
]

DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_learning_rate_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": "Baseline LR 1e-3",
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "learning_rate": 1e-3,
    "learning_rate_schedule": "constant",
    "learning_rate_end": 1e-3,
    "learning_rate_decay_rate": 1.0,
    "learning_rate_warmup_iterations": 0,
})
