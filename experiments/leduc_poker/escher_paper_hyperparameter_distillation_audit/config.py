"""Frozen contract for Experiment 46."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    ARMS,
    ARM_ORDER,
    DEFAULT_CONFIG as EXPERIMENT_45_CONFIG,
    DEFAULT_SEEDS,
    GROUPED_CE,
    GROUPED_CE_4X,
    ROW_CE,
    ROW_MSE,
    SAFETY_MAX_ITERATIONS,
    TRAINING_WALL_CLOCK_SECONDS,
    validate_config as validate_experiment_45_config,
)


EXPERIMENT_ID = 46
EXPERIMENT_NAME = "leduc_poker_escher_paper_hyperparameter_distillation_audit"

PAPER_HYPERPARAMETERS = {
    "num_traversals": 1_000,
    "num_val_fn_traversals": 1_000,
    "batch_size_regret": 2_048,
    "batch_size_value": 2_048,
    "regret_network_train_steps": 5_000,
    "value_network_train_steps": 5_000,
    "policy_network_train_steps": 10_000,
}

DEFAULT_CONFIG = deepcopy(EXPERIMENT_45_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": EXPERIMENT_NAME,
    **PAPER_HYPERPARAMETERS,
})


def validate_config(config: Mapping[str, object], *, smoke: bool = False) -> None:
    """Preserve Experiment 45 and change only the reported paper parameters."""
    validate_experiment_45_config(config, smoke=smoke)
    if not smoke:
        for key, expected in PAPER_HYPERPARAMETERS.items():
            if int(config[key]) != int(expected):
                raise ValueError(
                    f"Experiment 46 requires {key}={expected}, got {config[key]}"
                )
    if str(config["experiment_name"]) != EXPERIMENT_NAME:
        raise ValueError("Experiment 46 has the wrong experiment_name")


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPERIMENT_45_CONFIG",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GROUPED_CE",
    "GROUPED_CE_4X",
    "PAPER_HYPERPARAMETERS",
    "ROW_CE",
    "ROW_MSE",
    "SAFETY_MAX_ITERATIONS",
    "TRAINING_WALL_CLOCK_SECONDS",
    "validate_config",
]

