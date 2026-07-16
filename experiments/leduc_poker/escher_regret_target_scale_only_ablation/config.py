"""Experiment 36 configuration derived from the Experiment 28 candidate."""

from __future__ import annotations

from copy import deepcopy

from escher_poker.regret_target_processing import (
    BATCH_RMS,
    BATCH_STANDARDIZE,
    EMA_STD,
    FIXED_UTILITY_SCALE,
    RAW,
)
from escher_poker.regret_targets import (
    AUTHOR_STATE_VALUE,
    PAPER_POLICY_WEIGHTED_Q,
)
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)


DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

# OpenSpiel Leduc has utilities in [-13, 13]. Regrets can span that complete
# game-wide range, so the fixed scale is 13 - (-13) = 26.
LEDUC_UTILITY_RANGE = 26.0
DEFAULT_EMA_DECAY = 0.99

BASELINE_VARIANT_ID = "experiment_28_batch_centered_baseline"
CORRECTED_STANDARDIZED_CONTROL_ID = "corrected_batch_centered_control"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    regret_target_baseline,
    regret_target_processing,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "regret_target_baseline": regret_target_baseline,
        "regret_target_processing": regret_target_processing,
        "regret_target_fixed_scale": LEDUC_UTILITY_RANGE,
        "regret_target_ema_decay": DEFAULT_EMA_DECAY,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Experiment 28 batch-centred baseline",
        (
            "Exact Experiment 28 author-code target with minibatch mean "
            "subtraction and minibatch standard-deviation scaling."
        ),
        AUTHOR_STATE_VALUE,
        BATCH_STANDARDIZE,
    ),
    _variant(
        CORRECTED_STANDARDIZED_CONTROL_ID,
        "Corrected + batch-centred control",
        (
            "Policy-weighted-Q corrected target with Experiment 28's minibatch "
            "mean subtraction and standard-deviation scaling."
        ),
        PAPER_POLICY_WEIGHTED_Q,
        BATCH_STANDARDIZE,
    ),
    _variant(
        "corrected_raw",
        "Corrected + raw targets",
        "Policy-weighted-Q corrected target with no supervised-target processing.",
        PAPER_POLICY_WEIGHTED_Q,
        RAW,
    ),
    _variant(
        "corrected_fixed_utility_scale",
        "Corrected + fixed utility scale",
        (
            "Policy-weighted-Q corrected target divided by Leduc's fixed "
            "game-wide utility range of 26."
        ),
        PAPER_POLICY_WEIGHTED_Q,
        FIXED_UTILITY_SCALE,
    ),
    _variant(
        "corrected_batch_rms",
        "Corrected + minibatch RMS",
        (
            "Policy-weighted-Q corrected target divided by the legal-target "
            "minibatch RMS, without mean subtraction."
        ),
        PAPER_POLICY_WEIGHTED_Q,
        BATCH_RMS,
    ),
    _variant(
        "corrected_persistent_std",
        "Corrected + persistent std",
        (
            "Policy-weighted-Q corrected target divided by an EMA standard "
            "deviation, without subtracting the tracked mean."
        ),
        PAPER_POLICY_WEIGHTED_Q,
        EMA_STD,
    ),
]


DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_target_scale_only_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "regret_target_baseline": AUTHOR_STATE_VALUE,
    "regret_target_processing": BATCH_STANDARDIZE,
    "regret_target_fixed_scale": LEDUC_UTILITY_RANGE,
    "regret_target_ema_decay": DEFAULT_EMA_DECAY,
})
