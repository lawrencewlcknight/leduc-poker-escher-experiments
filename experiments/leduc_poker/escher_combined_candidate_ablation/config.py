"""Configuration for Experiment 41's combined ESCHER candidate ablation."""

from __future__ import annotations

from copy import deepcopy

from escher_poker.regret_target_processing import BATCH_STANDARDIZE
from escher_poker.regret_targets import AUTHOR_STATE_VALUE, PAPER_POLICY_WEIGHTED_Q
from escher_poker.replay import INFOSET_STRATIFIED, RESERVOIR
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
BASELINE_VARIANT_ID = "experiment_28_baseline"
CANDIDATE_VARIANT_ID = "policy_q_stratified_uniform"
MAXIMUM_STACK_VARIANT_ID = "policy_q_stratified_exact_balanced"


def _variant(
    variant_id,
    variant_label,
    variant_description,
    *,
    regret_target_baseline,
    regret_replay_mode,
    use_balanced_probs,
    balanced_sampling_mix,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "regret_target_baseline": regret_target_baseline,
        "regret_target_processing": BATCH_STANDARDIZE,
        "regret_replay_mode": regret_replay_mode,
        "use_balanced_probs": bool(use_balanced_probs),
        "balanced_sampling_mix": float(balanced_sampling_mix),
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Experiment 28 baseline",
        "Exact Experiment 28 target, replay, and uniform fixed sampling.",
        regret_target_baseline=AUTHOR_STATE_VALUE,
        regret_replay_mode=RESERVOIR,
        use_balanced_probs=False,
        balanced_sampling_mix=0.0,
    ),
    _variant(
        CANDIDATE_VARIANT_ID,
        "Policy Q + stratified replay + uniform sampling",
        (
            "Policy-weighted-Q targets with Experiment 28 batch-centred "
            "standardization, infoset-stratified regret replay, and uniform "
            "fixed sampling."
        ),
        regret_target_baseline=PAPER_POLICY_WEIGHTED_Q,
        regret_replay_mode=INFOSET_STRATIFIED,
        use_balanced_probs=False,
        balanced_sampling_mix=0.0,
    ),
    _variant(
        MAXIMUM_STACK_VARIANT_ID,
        "Policy Q + stratified replay + exact balanced sampling",
        (
            "Maximum stack: the evidence-weighted candidate plus exact "
            "leaf-balanced fixed sampling."
        ),
        regret_target_baseline=PAPER_POLICY_WEIGHTED_Q,
        regret_replay_mode=INFOSET_STRATIFIED,
        use_balanced_probs=True,
        balanced_sampling_mix=1.0,
    ),
]


DEFAULT_CONFIG = deepcopy(EXPERIMENT_28_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_combined_candidate_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "execution_backend": "sequential",
    "regret_target_baseline": AUTHOR_STATE_VALUE,
    "regret_target_processing": BATCH_STANDARDIZE,
    "regret_replay_mode": RESERVOIR,
    "use_balanced_probs": False,
    "balanced_sampling_mix": 0.0,
    "track_sampling_coverage": False,
})
