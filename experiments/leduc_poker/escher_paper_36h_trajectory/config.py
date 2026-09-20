"""Frozen scientific contract for Experiment 49."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from experiments.leduc_poker.escher_final_candidate_trajectory_15m.config import (
    DEFAULT_SEEDS as FIVE_SEED_COMPARISON_SEEDS,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    ARMS,
    GROUPED_CE,
)
from experiments.leduc_poker.escher_paper_hyperparameter_distillation_audit.config import (
    DEFAULT_CONFIG as EXPERIMENT_46_CONFIG,
)
from experiments.leduc_poker.escher_paper_hyperparameter_distillation_audit.config import (
    PAPER_HYPERPARAMETERS,
)

EXPERIMENT_ID = 49
EXPERIMENT_NAME = "leduc_poker_escher_paper_36h_trajectory"
DEFAULT_SEEDS = list(FIVE_SEED_COMPARISON_SEEDS)
TRAINING_WALL_CLOCK_SECONDS = 36 * 60 * 60
CHECKPOINT_INTERVAL_SECONDS = 30 * 60
TARGET_NODES_TOUCHED = 15_000_000
SELECTED_POLICY_ARM = GROUPED_CE
SELECTED_POLICY_SPEC = deepcopy(ARMS[SELECTED_POLICY_ARM])

DEFAULT_CONFIG = deepcopy(EXPERIMENT_46_CONFIG)
DEFAULT_CONFIG.update(
    {
        "experiment_name": EXPERIMENT_NAME,
        "training_wall_clock_seconds": TRAINING_WALL_CLOCK_SECONDS,
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "selected_policy_arm": SELECTED_POLICY_ARM,
        "selected_policy_loss": SELECTED_POLICY_SPEC["loss"],
        "selected_policy_grouped": bool(SELECTED_POLICY_SPEC["grouped"]),
        "selected_policy_example_multiplier": int(SELECTED_POLICY_SPEC["example_multiplier"]),
        "source_policy_fit_mode": "deferred_grouped_soft_target_cross_entropy",
        "fit_final_policy_after_training": False,
        "exclude_checkpoint_time_from_training_budget": True,
        "checkpoint_directory": "frozen_checkpoints",
        "checkpoint_manifest_filename": "checkpoint_manifest.csv",
        "checkpoint_metrics_filename": "checkpoint_metrics.csv",
        "training_progress_every": 10,
        "compute_exploitability": False,
    }
)


def validate_config(config: Mapping[str, object], *, smoke: bool = False) -> None:
    """Reject accidental changes to the production Experiment 49 contract."""
    if str(config["experiment_name"]) != EXPERIMENT_NAME:
        raise ValueError("Experiment 49 has the wrong experiment_name")
    if not smoke:
        if list(DEFAULT_SEEDS) != [1234, 2025, 31415, 27182, 16180]:
            raise ValueError("Experiment 49 must use the five thesis-comparison seeds")
        if float(config["training_wall_clock_seconds"]) != TRAINING_WALL_CLOCK_SECONDS:
            raise ValueError("Experiment 49 requires 36 active training hours")
        if float(config["checkpoint_interval_seconds"]) != CHECKPOINT_INTERVAL_SECONDS:
            raise ValueError("Experiment 49 requires 30-minute checkpoints")
        if int(config["target_nodes_touched"]) != TARGET_NODES_TOUCHED:
            raise ValueError("Experiment 49 requires a 15-million-node endpoint")
        for key, expected in PAPER_HYPERPARAMETERS.items():
            if int(config[key]) != int(expected):
                raise ValueError(f"Experiment 49 requires paper parameter {key}={expected}")
        if int(config["average_policy_memory_capacity"]) != 1_000_000:
            raise ValueError("Experiment 49 requires the one-million-row policy reservoir")
    if float(config["training_wall_clock_seconds"]) <= 0.0:
        raise ValueError("training_wall_clock_seconds must be positive")
    if float(config["checkpoint_interval_seconds"]) <= 0.0:
        raise ValueError("checkpoint_interval_seconds must be positive")
    if int(config["target_nodes_touched"]) <= 0:
        raise ValueError("target_nodes_touched must be positive")
    if bool(config["compute_exploitability"]):
        raise ValueError("Exploitability must be deferred until after training")
    if bool(config["fit_final_policy_after_training"]):
        raise ValueError("The source worker must not fit an in-loop/final policy")
    if not bool(config["exclude_checkpoint_time_from_training_budget"]):
        raise ValueError("Checkpoint I/O must be excluded from active training time")
    if str(config["selected_policy_arm"]) != SELECTED_POLICY_ARM:
        raise ValueError("Experiment 49 must use the selected grouped-CE policy fit")
    if not bool(config["selected_policy_grouped"]):
        raise ValueError("Experiment 49's selected policy fit must be grouped")


__all__ = [
    "CHECKPOINT_INTERVAL_SECONDS",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPERIMENT_46_CONFIG",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "PAPER_HYPERPARAMETERS",
    "SELECTED_POLICY_ARM",
    "SELECTED_POLICY_SPEC",
    "TARGET_NODES_TOUCHED",
    "TRAINING_WALL_CLOCK_SECONDS",
    "validate_config",
]
