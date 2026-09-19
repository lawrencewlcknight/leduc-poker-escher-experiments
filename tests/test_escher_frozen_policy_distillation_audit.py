"""Contract and mathematical tests for Experiment 45."""

from argparse import Namespace

import numpy as np
import pyspiel

from escher_poker.experiment_utils import make_escher_solver
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    ARM_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.distillation import (
    FrozenReservoir,
    cross_entropy_objective_equivalence,
    group_reservoir,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.run import (
    build_config,
)


def test_experiment_45_preserves_core_memories_and_enlarges_only_policy():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert DEFAULT_CONFIG["num_iterations"] == 1_300
    assert DEFAULT_CONFIG["check_exploitability_every"] == 10
    assert DEFAULT_CONFIG["memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["average_policy_memory_capacity"] == 1_000_000
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 256, 128)
    assert tuple(DEFAULT_CONFIG["distillation_arms"]) == ARM_ORDER


def test_grouped_cross_entropy_is_exact_row_objective():
    reservoir = FrozenReservoir(
        info_states=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        action_probs=np.asarray(
            [[0.8, 0.2], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32
        ),
        iterations=np.asarray([1, 3, 2], dtype=np.float32),
        legal_actions=np.ones((3, 2), dtype=np.float32),
        reach_probs=np.ones(3, dtype=np.float32),
        obs_indices=np.arange(3, dtype=np.float32),
    )
    grouped = group_reservoir(reservoir, iteration=4)
    predictions = np.asarray([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
    row, compressed = cross_entropy_objective_equivalence(
        reservoir, grouped, predictions, iteration=4
    )
    assert np.isclose(row, compressed, atol=1e-7)


def test_smoke_config_reduces_all_expensive_dimensions():
    args = Namespace(
        iterations=None,
        traversals=None,
        value_traversals=None,
        evaluation_interval=None,
        memory_capacity=None,
        average_policy_memory_capacity=None,
        policy_network_train_steps=None,
        regret_network_train_steps=None,
        value_network_train_steps=None,
        batch_size_regret=None,
        batch_size_value=None,
        batch_size_average_policy=None,
        policy_network_layers=None,
        regret_network_layers=None,
        value_network_layers=None,
        reservoir_decode_chunk_size=None,
        smoke=True,
    )
    config = build_config(args)
    assert config["num_iterations"] == 2
    assert config["average_policy_memory_capacity"] == 256
    assert config["policy_network_layers"] == (8, 8)
    assert config["policy_network_train_steps"] == 2


def test_solver_uses_independent_policy_and_core_memory_capacities():
    config = dict(DEFAULT_CONFIG)
    config.update({
        "num_iterations": 0,
        "memory_capacity": 7,
        "average_policy_memory_capacity": 19,
        "policy_network_layers": (8, 8),
        "regret_network_layers": (8, 8),
        "value_network_layers": (8, 8),
        "regret_network_head_units": 4,
    })
    solver = make_escher_solver(pyspiel.load_game("leduc_poker"), config)
    assert solver._average_policy_memories._reservoir_buffer_capacity == 19
    assert solver._regret_memories[0]._reservoir_buffer_capacity == 7
    assert solver._regret_memories[1]._reservoir_buffer_capacity == 7
    assert solver._value_memory._reservoir_buffer_capacity == 7
    assert solver._value_memory_test._reservoir_buffer_capacity == 7
