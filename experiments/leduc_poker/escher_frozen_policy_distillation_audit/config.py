"""Frozen contract for Experiment 45."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.leduc_poker.escher_final_candidate_trajectory_15m.config import (
    DEFAULT_CONFIG as EXPERIMENT_44_CONFIG,
)


EXPERIMENT_ID = 45
EXPERIMENT_NAME = "leduc_poker_escher_frozen_policy_distillation_audit"
DEFAULT_SEEDS = [1234, 2025, 31415]
TRAINING_WALL_CLOCK_SECONDS = 12 * 60 * 60
SAFETY_MAX_ITERATIONS = 100_000
TRAINING_PROGRESS_EVERY = 10

ROW_MSE = "row_mse_current_budget"
ROW_CE = "row_soft_target_ce_matched_examples"
GROUPED_CE = "grouped_soft_target_ce_matched_examples"
GROUPED_CE_4X = "grouped_soft_target_ce_4x_extended"
ARM_ORDER = (ROW_MSE, ROW_CE, GROUPED_CE, GROUPED_CE_4X)
ARMS = {
    ROW_MSE: {
        "label": "Row-wise MSE (current budget)",
        "loss": "mse",
        "grouped": False,
        "example_multiplier": 1,
    },
    ROW_CE: {
        "label": "Row-wise soft-target CE (matched examples)",
        "loss": "soft_target_cross_entropy",
        "grouped": False,
        "example_multiplier": 1,
    },
    GROUPED_CE: {
        "label": "Grouped soft-target CE (matched examples)",
        "loss": "soft_target_cross_entropy",
        "grouped": True,
        "example_multiplier": 1,
    },
    GROUPED_CE_4X: {
        "label": "Grouped soft-target CE (4x fitting)",
        "loss": "soft_target_cross_entropy",
        "grouped": True,
        "example_multiplier": 4,
    },
}

DEFAULT_CONFIG = deepcopy(EXPERIMENT_44_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": EXPERIMENT_NAME,
    # Time, rather than iterations or touched nodes, defines the endpoint. The
    # solver completes an in-flight iteration after the boundary; this large
    # iteration cap is only a guard against an unavailable monotonic clock.
    "training_wall_clock_seconds": TRAINING_WALL_CLOCK_SECONDS,
    "num_iterations": SAFETY_MAX_ITERATIONS,
    "expected_final_nodes_touched": None,
    # Intermediate average-policy fitting is deliberately disabled.  The full
    # twelve-hour budget is reserved for regret/value learning; the playable
    # source policy is fitted once after the timed loop.  Cheap learner-only
    # telemetry is retained independently at this cadence.
    "check_exploitability_every": 10,
    "compute_exploitability": False,
    "training_progress_every": TRAINING_PROGRESS_EVERY,
    "source_policy_fit_mode": "final_only_after_timed_training",
    "memory_capacity": 50_000,
    "regret_memory_capacity": 50_000,
    "value_memory_capacity": 50_000,
    "value_validation_memory_capacity": 50_000,
    "average_policy_memory_capacity": 1_000_000,
    "policy_network_layers": (256, 256, 128),
    "save_final_checkpoints": False,
    "save_policy_weights": False,
    "frozen_reservoir_filename": "frozen_average_policy_reservoir.npz",
    "reservoir_decode_chunk_size": 25_000,
    "distillation_arms": ARM_ORDER,
    "fit_seed_offset": 450_000,
    "matched_example_definition": (
        "actual network input rows presented to the optimiser"
    ),
})


def validate_common_config(config: Mapping[str, object], *, smoke: bool = False) -> None:
    """Validate the shared frozen-reservoir audit design."""
    if not smoke and int(config["average_policy_memory_capacity"]) != 1_000_000:
        raise ValueError("Experiment 45 requires a 1,000,000-row policy reservoir")
    if not smoke and tuple(config["policy_network_layers"]) != (256, 256, 128):
        raise ValueError("The audit fixes the policy network at 256x256x128")
    if (
        not smoke
        and float(config["training_wall_clock_seconds"])
        != float(TRAINING_WALL_CLOCK_SECONDS)
    ):
        raise ValueError("Experiment 45 requires exactly 12 active training hours")
    if float(config["training_wall_clock_seconds"]) <= 0.0:
        raise ValueError("training_wall_clock_seconds must be positive")
    if bool(config.get("compute_exploitability", True)):
        raise ValueError(
            "Experiments 45--48 reserve the timed budget for source learning; "
            "intermediate average-policy fitting must remain disabled"
        )
    if int(config.get("training_progress_every", 0)) <= 0:
        raise ValueError("training_progress_every must be positive")
    if config.get("source_policy_fit_mode") != "final_only_after_timed_training":
        raise ValueError("The source policy must be fitted only after timed training")
    if not smoke and config.get("expected_final_nodes_touched") is not None:
        raise ValueError("Experiment 45 has no fixed touched-node endpoint")
    if int(config["num_iterations"]) <= 0:
        raise ValueError("num_iterations must be positive")
    if int(config["policy_network_train_steps"]) <= 0:
        raise ValueError("policy_network_train_steps must be positive")
    if int(config["batch_size_average_policy"]) <= 0:
        raise ValueError("batch_size_average_policy must be positive")
    for name in (
        "memory_capacity",
        "regret_memory_capacity",
        "value_memory_capacity",
        "value_validation_memory_capacity",
        "average_policy_memory_capacity",
    ):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if tuple(config["distillation_arms"]) != ARM_ORDER:
        raise ValueError("All four frozen-reservoir arms are required")


def validate_config(config: Mapping[str, object], *, smoke: bool = False) -> None:
    validate_common_config(config, smoke=smoke)
    if not smoke:
        for name in (
            "memory_capacity",
            "regret_memory_capacity",
            "value_memory_capacity",
            "value_validation_memory_capacity",
        ):
            if int(config[name]) != 50_000:
                raise ValueError(
                    f"Experiment 45 requires {name}=50,000, got {config[name]}"
                )


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPERIMENT_44_CONFIG",
    "GROUPED_CE",
    "GROUPED_CE_4X",
    "ROW_CE",
    "ROW_MSE",
    "SAFETY_MAX_ITERATIONS",
    "TRAINING_PROGRESS_EVERY",
    "TRAINING_WALL_CLOCK_SECONDS",
    "validate_common_config",
    "validate_config",
]
