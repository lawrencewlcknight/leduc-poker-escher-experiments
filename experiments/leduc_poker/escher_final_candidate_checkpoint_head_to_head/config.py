"""Configuration for Experiment 43: ESCHER checkpoint head-to-head."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
    DEFAULT_SEEDS as EXPERIMENT_28_SEEDS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


DEFAULT_SEEDS = list(EXPERIMENT_28_SEEDS)

# ESCHERSolver.solve executes range(num_iterations + 1). Scaling the completed
# long-horizon baseline's 9,231,941 mean nodes at 801 passes by 1,301 / 801
# gives 14,994,701 expected nodes. The configured indices are the 20% temporal
# milestones.
TARGET_NUM_ITERATIONS = 1_300
TARGET_SOLVE_PASSES = TARGET_NUM_ITERATIONS + 1
CHECKPOINT_SCHEDULE = (260, 520, 780, 1_040, 1_300)
EXPECTED_FINAL_NODES_TOUCHED = 15_000_000

DEFAULT_CONFIG = deepcopy(EXPERIMENT_28_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_final_candidate_checkpoint_head_to_head",
    "num_iterations": TARGET_NUM_ITERATIONS,
    "checkpoint_schedule": CHECKPOINT_SCHEDULE,
    "target_solve_pass_count": TARGET_SOLVE_PASSES,
    "expected_final_nodes_touched": EXPECTED_FINAL_NODES_TOUCHED,
    "equivalence_epsilon": 1e-3,
    "temporal_x_axis": "nodes_touched",
    "require_complete_checkpoint_schedule": True,
    "annotate_heatmap": True,
    # Leduc permits exact expected-value evaluation, so no Monte Carlo match
    # count is needed. Independent training seeds are the inferential unit.
    "head_to_head_evaluation": "exact_open_spiel_two_seat",
    "run_monte_carlo_validation": False,
    "num_mc_episodes": 0,
    "save_policy_weights": False,
})
DEFAULT_CONFIG = make_variant_config(DEFAULT_CONFIG, {})


def validate_config(config: Mapping[str, object]) -> None:
    """Ensure every requested snapshot follows a fresh average-policy fit."""
    schedule = tuple(int(value) for value in config["checkpoint_schedule"])
    num_iterations = int(config["num_iterations"])
    interval = int(config["check_exploitability_every"])
    if not schedule or any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("checkpoint_schedule must be non-empty and strictly increasing")
    if any(checkpoint <= 0 for checkpoint in schedule):
        raise ValueError("checkpoint_schedule entries must be positive")
    if interval <= 0:
        raise ValueError("check_exploitability_every must be positive")
    if schedule[-1] != num_iterations:
        raise ValueError(
            "The final checkpoint must equal num_iterations so every run ends "
            "at the configured training budget"
        )
    if not bool(config.get("compute_exploitability", False)):
        raise ValueError("compute_exploitability must be true to fit checkpoint policies")
    stale = [checkpoint for checkpoint in schedule if checkpoint % interval != 0]
    if stale:
        raise ValueError(
            "Every checkpoint must coincide with average-policy evaluation; "
            f"incompatible checkpoints: {stale}"
        )


__all__ = [
    "CHECKPOINT_SCHEDULE",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPECTED_FINAL_NODES_TOUCHED",
    "EXPERIMENT_28_CONFIG",
    "TARGET_NUM_ITERATIONS",
    "TARGET_SOLVE_PASSES",
    "validate_config",
]
