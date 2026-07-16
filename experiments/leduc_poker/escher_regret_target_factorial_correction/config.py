"""Configuration for Experiment 37's staged 2x2 correction factorial."""

from __future__ import annotations

from copy import deepcopy

from escher_poker.regret_target_processing import BATCH_RMS, BATCH_STANDARDIZE
from escher_poker.regret_targets import AUTHOR_STATE_VALUE, PAPER_POLICY_WEIGHTED_Q
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
)


SCREENING_SEEDS = [1234, 2025, 31415]
CONFIRMATION_SEEDS = [27182, 16180, 4242, 8675309, 7]
TARGET_NODES = 1_000_000.0
MEANINGFUL_SUCCESS_THRESHOLD = 0.3
CONFIRMATION_TOP_K = 2

BASELINE_VARIANT_ID = "current_baseline_current_standardization"
POLICY_ONLY_VARIANT_ID = "policy_weighted_q_baseline_only"
SCALE_ONLY_VARIANT_ID = "scale_only_target_normalization_only"
BOTH_VARIANT_ID = "both_corrections"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    *,
    policy_weighted_q_correction,
    scale_only_normalization,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_weighted_q_correction": bool(policy_weighted_q_correction),
        "scale_only_normalization": bool(scale_only_normalization),
        "regret_target_baseline": (
            PAPER_POLICY_WEIGHTED_Q
            if policy_weighted_q_correction
            else AUTHOR_STATE_VALUE
        ),
        "regret_target_processing": (
            BATCH_RMS if scale_only_normalization else BATCH_STANDARDIZE
        ),
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Current baseline + current standardization",
        "Exact Experiment 28 target definition and batch-centred standardization.",
        policy_weighted_q_correction=False,
        scale_only_normalization=False,
    ),
    _variant(
        POLICY_ONLY_VARIANT_ID,
        "Policy-weighted Q baseline only",
        (
            "Policy-weighted-Q regret target with Experiment 28's batch-centred "
            "standardization retained."
        ),
        policy_weighted_q_correction=True,
        scale_only_normalization=False,
    ),
    _variant(
        SCALE_ONLY_VARIANT_ID,
        "Scale-only target normalization only",
        (
            "Experiment 28's author-code regret target divided by legal-target "
            "minibatch RMS, without mean subtraction."
        ),
        policy_weighted_q_correction=False,
        scale_only_normalization=True,
    ),
    _variant(
        BOTH_VARIANT_ID,
        "Both corrections",
        (
            "Policy-weighted-Q regret target divided by legal-target minibatch "
            "RMS, without mean subtraction."
        ),
        policy_weighted_q_correction=True,
        scale_only_normalization=True,
    ),
]


DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_target_factorial_correction",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "factorial_variants": tuple(VARIANTS),
    "policy_weighted_q_correction": False,
    "scale_only_normalization": False,
    "regret_target_baseline": AUTHOR_STATE_VALUE,
    "regret_target_processing": BATCH_STANDARDIZE,
    "target_nodes": TARGET_NODES,
    "meaningful_success_threshold": MEANINGFUL_SUCCESS_THRESHOLD,
    "confirmation_top_k": CONFIRMATION_TOP_K,
})
