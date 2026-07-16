"""Configuration for Experiment 42's 20x-node candidate ablation."""

from __future__ import annotations

from copy import deepcopy

from escher_poker.regret_target_processing import BATCH_STANDARDIZE
from escher_poker.regret_targets import AUTHOR_STATE_VALUE, PAPER_POLICY_WEIGHTED_Q
from escher_poker.replay import INFOSET_STRATIFIED, RESERVOIR
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
BASELINE_VARIANT_ID = "experiment_28_20x_nodes"
CANDIDATE_VARIANT_ID = "policy_q_stratified_uniform_20x_nodes"
SOLVE_PASS_MULTIPLIER = 20
EXPERIMENT_28_SOLVE_PASSES = int(EXPERIMENT_28_CONFIG["num_iterations"]) + 1
LONG_RUN_SOLVE_PASSES = SOLVE_PASS_MULTIPLIER * EXPERIMENT_28_SOLVE_PASSES
LONG_RUN_NUM_ITERATIONS = LONG_RUN_SOLVE_PASSES - 1


def _variant(
    variant_id,
    variant_label,
    variant_description,
    *,
    regret_target_baseline,
    regret_replay_mode,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "regret_target_baseline": regret_target_baseline,
        "regret_target_processing": BATCH_STANDARDIZE,
        "regret_replay_mode": regret_replay_mode,
        "use_balanced_probs": False,
        "balanced_sampling_mix": 0.0,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Experiment 28 baseline (20x solve passes)",
        (
            "Exact Experiment 28 algorithm trained for 20 times its original "
            "number of solve passes."
        ),
        regret_target_baseline=AUTHOR_STATE_VALUE,
        regret_replay_mode=RESERVOIR,
    ),
    _variant(
        CANDIDATE_VARIANT_ID,
        "Policy Q + stratified replay (20x solve passes)",
        (
            "Policy-weighted-Q targets, Experiment 28 batch-centred "
            "standardization, infoset-stratified regret replay, and uniform "
            "fixed sampling, trained for 20 times the Experiment 28 passes."
        ),
        regret_target_baseline=PAPER_POLICY_WEIGHTED_Q,
        regret_replay_mode=INFOSET_STRATIFIED,
    ),
]


DEFAULT_CONFIG = deepcopy(EXPERIMENT_28_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_long_horizon_candidate_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "execution_backend": "sequential",
    "num_iterations": LONG_RUN_NUM_ITERATIONS,
    "solve_pass_multiplier": SOLVE_PASS_MULTIPLIER,
    "experiment_28_solve_passes": EXPERIMENT_28_SOLVE_PASSES,
    "long_run_solve_passes": LONG_RUN_SOLVE_PASSES,
    "regret_target_baseline": AUTHOR_STATE_VALUE,
    "regret_target_processing": BATCH_STANDARDIZE,
    "regret_replay_mode": RESERVOIR,
    "use_balanced_probs": False,
    "balanced_sampling_mix": 0.0,
    "track_sampling_coverage": False,
})
