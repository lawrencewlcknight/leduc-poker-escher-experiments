"""Frozen contract for Experiment 47."""

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
    validate_common_config,
)


EXPERIMENT_ID = 47
EXPERIMENT_NAME = "leduc_poker_escher_large_regret_reservoir_distillation_audit"

DEFAULT_CONFIG = deepcopy(EXPERIMENT_45_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": EXPERIMENT_NAME,
    "regret_memory_capacity": 1_000_000,
})


def validate_config(config: Mapping[str, object], *, smoke: bool = False) -> None:
    """Require Experiment 45 with only the regret capacities enlarged."""
    validate_common_config(config, smoke=smoke)
    if not smoke:
        expected = {
            "memory_capacity": 50_000,
            "regret_memory_capacity": 1_000_000,
            "value_memory_capacity": 50_000,
            "value_validation_memory_capacity": 50_000,
            "average_policy_memory_capacity": 1_000_000,
        }
        for key, value in expected.items():
            if int(config[key]) != value:
                raise ValueError(
                    f"Experiment 47 requires {key}={value:,}, got {config[key]}"
                )
    if str(config["experiment_name"]) != EXPERIMENT_NAME:
        raise ValueError("Experiment 47 has the wrong experiment_name")


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
    "ROW_CE",
    "ROW_MSE",
    "SAFETY_MAX_ITERATIONS",
    "TRAINING_WALL_CLOCK_SECONDS",
    "validate_config",
]
