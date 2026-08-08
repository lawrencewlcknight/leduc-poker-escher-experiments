"""Configuration for Experiment 44: dense 15M-node ESCHER trajectory."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.config import (
    DEFAULT_CONFIG as EXPERIMENT_43_CONFIG,
    DEFAULT_SEEDS as EXPERIMENT_43_SEEDS,
    EXPECTED_FINAL_NODES_TOUCHED,
    TARGET_NUM_ITERATIONS,
    TARGET_SOLVE_PASSES,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


DEFAULT_SEEDS = list(EXPERIMENT_43_SEEDS)
DEFAULT_CONFIG = deepcopy(EXPERIMENT_43_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_final_candidate_trajectory_15m",
    "trajectory_history_filename": "trajectory_history.csv",
    "trajectory_summary_filename": "trajectory_summary.csv",
    "trajectory_alignment_key": "checkpoint_index",
    "trajectory_x_axis": "nodes_touched",
    "record_initial_policy_evaluation": True,
    "trajectory_auc_start_nodes": 0,
    "trajectory_auc_end_nodes": EXPECTED_FINAL_NODES_TOUCHED,
    "trajectory_auc_hold_boundaries": True,
    "final_window_width_nodes": 1_000_000,
})
DEFAULT_CONFIG = make_variant_config(DEFAULT_CONFIG, {})


def expected_trajectory_points(config: Mapping[str, object]) -> int:
    """Return the evaluations produced at loop indices 0, interval, ... ."""
    iterations = int(config["num_iterations"])
    interval = int(config["check_exploitability_every"])
    solver_points = iterations // interval + 1
    return solver_points + int(bool(config.get("record_initial_policy_evaluation", False)))


EXPECTED_TRAJECTORY_POINTS = expected_trajectory_points(DEFAULT_CONFIG)


def validate_config(config: Mapping[str, object]) -> None:
    """Validate the requirements for a dense comparison trajectory."""
    iterations = int(config["num_iterations"])
    interval = int(config["check_exploitability_every"])
    if iterations < 0:
        raise ValueError("num_iterations must be non-negative")
    if interval <= 0:
        raise ValueError("check_exploitability_every must be positive")
    if not bool(config.get("compute_exploitability", False)):
        raise ValueError("compute_exploitability must be true to save the trajectory")
    if not bool(config.get("record_initial_policy_evaluation", False)):
        raise ValueError(
            "record_initial_policy_evaluation must be true for 0-node AUC"
        )
    auc_start = float(config["trajectory_auc_start_nodes"])
    auc_end = config.get("trajectory_auc_end_nodes")
    if auc_start != 0:
        raise ValueError("trajectory_auc_start_nodes must be zero")
    if auc_end is not None and float(auc_end) <= auc_start:
        raise ValueError("trajectory_auc_end_nodes must be greater than zero")
    if float(config.get("final_window_width_nodes", 0)) <= 0:
        raise ValueError("final_window_width_nodes must be positive")
    if interval > iterations and iterations > 0:
        raise ValueError(
            "check_exploitability_every must not exceed num_iterations"
        )


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPECTED_FINAL_NODES_TOUCHED",
    "EXPECTED_TRAJECTORY_POINTS",
    "EXPERIMENT_43_CONFIG",
    "TARGET_NUM_ITERATIONS",
    "TARGET_SOLVE_PASSES",
    "expected_trajectory_points",
    "validate_config",
]
