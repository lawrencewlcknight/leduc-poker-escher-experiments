"""Frozen contract for Experiment 48."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.leduc_poker.escher_large_regret_reservoir_distillation_audit.config import (
    ARMS,
    ARM_ORDER,
    DEFAULT_CONFIG as EXPERIMENT_47_CONFIG,
    DEFAULT_SEEDS,
    GROUPED_CE,
    GROUPED_CE_4X,
    ROW_CE,
    ROW_MSE,
    SAFETY_MAX_ITERATIONS,
    TRAINING_WALL_CLOCK_SECONDS,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    validate_common_config,
)


EXPERIMENT_ID = 48
EXPERIMENT_NAME = "leduc_poker_escher_all_large_memory_distillation_audit"

DEFAULT_CONFIG = deepcopy(EXPERIMENT_47_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": EXPERIMENT_NAME,
    # Keep the backwards-compatible fallback aligned with every explicit
    # effective capacity so the persisted configuration is unambiguous.
    "memory_capacity": 1_000_000,
    "value_memory_capacity": 1_000_000,
    "value_validation_memory_capacity": 1_000_000,
})


def validate_config(config: Mapping[str, object], *, smoke: bool = False) -> None:
    """Require every effective reservoir and buffer capacity to be one million."""
    validate_common_config(config, smoke=smoke)
    if not smoke:
        for key in (
            "memory_capacity",
            "regret_memory_capacity",
            "value_memory_capacity",
            "value_validation_memory_capacity",
            "average_policy_memory_capacity",
        ):
            if int(config[key]) != 1_000_000:
                raise ValueError(
                    f"Experiment 48 requires {key}=1,000,000, got {config[key]}"
                )
    if str(config["experiment_name"]) != EXPERIMENT_NAME:
        raise ValueError("Experiment 48 has the wrong experiment_name")


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPERIMENT_47_CONFIG",
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
