"""Regression checks for memory-stable ESCHER network reinitialization."""

from copy import deepcopy

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
pyspiel = pytest.importorskip("pyspiel")

from escher_poker.experiment_utils import make_escher_solver  # noqa: E402
from escher_poker.seeding import set_seed_tf  # noqa: E402
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (  # noqa: E402
    DEFAULT_CONFIG,
)


def _small_solver():
    config = deepcopy(DEFAULT_CONFIG)
    config.update({
        "num_iterations": 0,
        "num_traversals": 1,
        "num_val_fn_traversals": 1,
        "value_test_traversals": 1,
        "memory_capacity": 8,
        "batch_size_regret": 2,
        "batch_size_value": 2,
        "batch_size_average_policy": 2,
        "policy_network_layers": (8, 4),
        "regret_network_layers": (8, 4),
        "value_network_layers": (8, 4),
        "regret_network_head_units": 4,
    })
    set_seed_tf(1234)
    return make_escher_solver(pyspiel.load_game(config["game_name"]), config)


def _weights_changed(before, after):
    return any(not np.array_equal(left, right) for left, right in zip(before, after))


def test_reinitialization_reuses_models_optimizers_and_train_graphs():
    solver = _small_solver()

    policy_model = solver._policy_network
    policy_optimizer = solver._optimizer_policy
    policy_train_step = solver._policy_train_step
    policy_weights = policy_model.get_weights()
    solver._reinitialize_policy_network()
    assert solver._policy_network is policy_model
    assert solver._optimizer_policy is policy_optimizer
    assert solver._policy_train_step is policy_train_step
    assert _weights_changed(policy_weights, policy_model.get_weights())

    regret_model = solver._regret_networks_train[0]
    regret_optimizer = solver._optimizer_regrets[0]
    regret_train_step = solver._regret_train_step[0]
    regret_weights = regret_model.get_weights()
    solver._reinitialize_regret_network(0)
    assert solver._regret_networks_train[0] is regret_model
    assert solver._optimizer_regrets[0] is regret_optimizer
    assert solver._regret_train_step[0] is regret_train_step
    assert _weights_changed(regret_weights, regret_model.get_weights())

    value_model = solver._val_network_train
    value_optimizer = solver._optimizer_value
    value_train_step = solver._value_train_step
    value_weights = value_model.get_weights()
    solver._reinitialize_value_network()
    assert solver._val_network_train is value_model
    assert solver._optimizer_value is value_optimizer
    assert solver._value_train_step is value_train_step
    assert _weights_changed(value_weights, value_model.get_weights())


def test_reinitialization_zeros_existing_adam_state():
    solver = _small_solver()
    optimizer = solver._optimizer_policy
    if hasattr(optimizer, "build"):
        optimizer.build(solver._policy_network.trainable_variables)
    optimizer_variables = optimizer.variables
    if callable(optimizer_variables):
        optimizer_variables = optimizer_variables()
    for variable in optimizer_variables:
        variable.assign(tf.ones_like(variable))

    solver._reinitialize_policy_network()

    for variable in optimizer_variables:
        assert np.allclose(variable.numpy(), 0.0)
