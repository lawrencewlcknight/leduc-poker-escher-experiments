"""Frozen-reservoir persistence, fitting and exact Leduc evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, expected_game_score

from escher_poker.constants import LEDUC_GAME_VALUE_PLAYER_0
from escher_poker.networks import PolicyNetwork
from escher_poker.seeding import set_seed_tf


@dataclass(frozen=True)
class FrozenReservoir:
    """Decoded final average-policy reservoir."""

    info_states: np.ndarray
    action_probs: np.ndarray
    iterations: np.ndarray
    legal_actions: np.ndarray
    reach_probs: np.ndarray
    obs_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.info_states))


@dataclass(frozen=True)
class GroupedReservoir:
    """Iteration-weighted sufficient statistics by information set."""

    info_states: np.ndarray
    action_probs: np.ndarray
    legal_actions: np.ndarray
    objective_weights: np.ndarray
    counts: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.info_states))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_serialized_reservoir(
    serialized_rows: Sequence[bytes],
    feature_description: Mapping,
    *,
    chunk_size: int = 25_000,
) -> FrozenReservoir:
    """Decode serialized TensorFlow examples without one giant parse tensor."""
    if not serialized_rows:
        raise ValueError("Cannot freeze an empty average-policy reservoir")
    decoded = {
        "info_state": [],
        "action_probs": [],
        "iteration": [],
        "legal_actions": [],
        "reach_prob": [],
        "obs_index": [],
    }
    for start in range(0, len(serialized_rows), int(chunk_size)):
        batch = tf.constant(serialized_rows[start:start + int(chunk_size)])
        parsed = tf.io.parse_example(batch, feature_description)
        for key in decoded:
            decoded[key].append(np.asarray(parsed[key].numpy(), dtype=np.float32))
    return FrozenReservoir(
        info_states=np.concatenate(decoded["info_state"], axis=0),
        action_probs=np.concatenate(decoded["action_probs"], axis=0),
        iterations=np.concatenate(decoded["iteration"], axis=0).reshape(-1),
        legal_actions=np.concatenate(decoded["legal_actions"], axis=0),
        reach_probs=np.concatenate(decoded["reach_prob"], axis=0).reshape(-1),
        obs_indices=np.concatenate(decoded["obs_index"], axis=0).reshape(-1),
    )


def save_frozen_reservoir(path: Path, reservoir: FrozenReservoir) -> dict:
    """Persist a lossless, compressed and reloadable reservoir snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        info_states=reservoir.info_states,
        action_probs=reservoir.action_probs,
        iterations=reservoir.iterations,
        legal_actions=reservoir.legal_actions,
        reach_probs=reservoir.reach_probs,
        obs_indices=reservoir.obs_indices,
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
        "num_rows": reservoir.size,
        "info_state_width": int(reservoir.info_states.shape[1]),
        "num_actions": int(reservoir.action_probs.shape[1]),
    }


def load_frozen_reservoir(path: Path) -> FrozenReservoir:
    with np.load(path, allow_pickle=False) as payload:
        return FrozenReservoir(
            info_states=np.asarray(payload["info_states"], dtype=np.float32),
            action_probs=np.asarray(payload["action_probs"], dtype=np.float32),
            iterations=np.asarray(payload["iterations"], dtype=np.float32),
            legal_actions=np.asarray(payload["legal_actions"], dtype=np.float32),
            reach_probs=np.asarray(payload["reach_probs"], dtype=np.float32),
            obs_indices=np.asarray(payload["obs_indices"], dtype=np.float32),
        )


def linear_iteration_weights(reservoir: FrozenReservoir, iteration: int) -> np.ndarray:
    if int(iteration) <= 0:
        raise ValueError("iteration must be positive")
    return np.asarray(
        reservoir.iterations * (2.0 / float(iteration)), dtype=np.float32
    )


def group_reservoir(
    reservoir: FrozenReservoir,
    *,
    iteration: int,
) -> GroupedReservoir:
    """Collapse repeated rows while preserving the row-wise CE objective."""
    weights = linear_iteration_weights(reservoir, iteration).astype(np.float64)
    unique, inverse, counts = np.unique(
        reservoir.info_states,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    masses = np.zeros(len(unique), dtype=np.float64)
    numerators = np.zeros(
        (len(unique), reservoir.action_probs.shape[1]), dtype=np.float64
    )
    np.add.at(masses, inverse, weights)
    np.add.at(
        numerators,
        inverse,
        reservoir.action_probs.astype(np.float64) * weights[:, None],
    )
    if np.any(masses <= 0.0):
        raise ValueError("Grouped policy target has non-positive objective mass")
    targets = numerators / masses[:, None]

    order = np.argsort(inverse, kind="stable")
    first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    masks = reservoir.legal_actions[order[first]]
    if not np.array_equal(reservoir.legal_actions, masks[inverse]):
        raise ValueError("Legal-action masks differ within an information set")

    # mean(group_weight * loss) == mean(row_weight * loss).
    objective_weights = masses * (float(len(unique)) / float(reservoir.size))
    return GroupedReservoir(
        info_states=np.asarray(unique, dtype=np.float32),
        action_probs=np.asarray(targets, dtype=np.float32),
        legal_actions=np.asarray(masks, dtype=np.float32),
        objective_weights=np.asarray(objective_weights, dtype=np.float32),
        counts=np.asarray(counts, dtype=np.int64),
    )


def _network(config: Mapping[str, object], input_size: int, num_actions: int):
    model = PolicyNetwork(
        input_size,
        tuple(config["policy_network_layers"]),
        num_actions,
        activation=str(config.get("policy_network_activation", "leakyrelu")),
        use_layer_norm=bool(config.get("policy_network_layer_norm", True)),
        residual_mode=str(config.get("policy_network_residual_mode", "same_width")),
        head_depth=int(config.get("policy_network_head_depth", 0)),
        head_units=config.get("policy_network_head_units"),
    )
    dummy_states = tf.zeros((1, input_size), dtype=tf.float32)
    dummy_masks = tf.ones((1, num_actions), dtype=tf.float32)
    model((dummy_states, dummy_masks), training=False)
    return model


def initial_policy_weights(
    config: Mapping[str, object],
    reservoir: FrozenReservoir,
    *,
    fit_seed: int,
) -> list[np.ndarray]:
    set_seed_tf(int(fit_seed))
    model = _network(
        config,
        reservoir.info_states.shape[1],
        reservoir.action_probs.shape[1],
    )
    return [np.asarray(weight).copy() for weight in model.get_weights()]


def policy_network_from_weights(
    config: Mapping[str, object],
    reservoir: FrozenReservoir,
    weights: Sequence[np.ndarray],
):
    """Build a policy network with explicit, already-materialised weights."""
    model = _network(
        config,
        reservoir.info_states.shape[1],
        reservoir.action_probs.shape[1],
    )
    model.set_weights(weights)
    return model


def _per_row_loss(targets, predictions, loss_name: str):
    if loss_name == "mse":
        return tf.reduce_mean(tf.square(targets - predictions), axis=-1)
    if loss_name == "soft_target_cross_entropy":
        predictions = tf.clip_by_value(predictions, 1e-12, 1.0)
        return -tf.reduce_sum(targets * tf.math.log(predictions), axis=-1)
    raise ValueError(f"Unsupported loss: {loss_name!r}")


def fit_policy(
    reservoir: FrozenReservoir,
    grouped: GroupedReservoir,
    config: Mapping[str, object],
    *,
    loss_name: str,
    use_grouped_data: bool,
    example_multiplier: int,
    fit_seed: int,
    base_weights: Sequence[np.ndarray],
) -> tuple[tf.keras.Model, dict]:
    """Fit one arm from a common initialization and frozen source."""
    set_seed_tf(int(fit_seed))
    model = _network(
        config,
        reservoir.info_states.shape[1],
        reservoir.action_probs.shape[1],
    )
    model.set_weights(base_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(config["learning_rate"]))
    current_steps = int(config["policy_network_train_steps"])
    current_batch = int(config["batch_size_average_policy"])
    target_examples = current_steps * current_batch * int(example_multiplier)

    @tf.function(reduce_retracing=True)
    def train_step(features, targets, masks, sample_weights):
        with tf.GradientTape() as tape:
            predictions = model((features, masks), training=True)
            losses = _per_row_loss(targets, predictions, loss_name)
            objective = tf.reduce_mean(losses * sample_weights)
        gradients = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return objective

    started = time.perf_counter()
    final_loss = math.nan
    if use_grouped_data:
        features = tf.constant(grouped.info_states)
        targets = tf.constant(grouped.action_probs)
        masks = tf.constant(grouped.legal_actions)
        weights = tf.constant(grouped.objective_weights)
        train_steps = int(math.ceil(target_examples / float(grouped.size)))
        for _ in range(train_steps):
            final_loss = float(train_step(features, targets, masks, weights).numpy())
        examples_processed = train_steps * grouped.size
    else:
        row_weights = linear_iteration_weights(
            reservoir, int(config["source_final_iteration"])
        )
        dataset = tf.data.Dataset.from_tensor_slices((
            reservoir.info_states,
            reservoir.action_probs,
            reservoir.legal_actions,
            row_weights,
        ))
        dataset = dataset.shuffle(
            min(reservoir.size, 1_000_000),
            seed=int(fit_seed),
            reshuffle_each_iteration=True,
        ).repeat().batch(current_batch).prefetch(1)
        train_steps = current_steps * int(example_multiplier)
        examples_processed = 0
        for features, targets, masks, weights in dataset.take(train_steps):
            final_loss = float(train_step(features, targets, masks, weights).numpy())
            examples_processed += int(features.shape[0])

    return model, {
        "fit_seconds": float(time.perf_counter() - started),
        "optimizer_steps": int(train_steps),
        "network_examples_processed": int(examples_processed),
        "target_network_examples": int(target_examples),
        "final_training_loss": float(final_loss),
    }


class _NeuralPolicy:
    def __init__(self, model):
        self.model = model

    def action_probabilities(self, state):
        legal = state.legal_actions()
        features = tf.constant(
            np.asarray(state.information_state_tensor(), dtype=np.float32)[None, :]
        )
        mask = np.zeros(state.num_distinct_actions(), dtype=np.float32)
        mask[legal] = 1.0
        probabilities = self.model(
            (features, tf.constant(mask[None, :])), training=False
        ).numpy()[0]
        return {action: float(probabilities[action]) for action in legal}


class _GroupedPolicy:
    def __init__(self, grouped: GroupedReservoir, num_actions: int):
        self.lookup = {
            np.asarray(row, dtype=np.float32).tobytes(): target
            for row, target in zip(grouped.info_states, grouped.action_probs)
        }
        self.num_actions = int(num_actions)
        self.missing_information_sets = set()

    def action_probabilities(self, state):
        legal = state.legal_actions()
        key = np.asarray(
            state.information_state_tensor(), dtype=np.float32
        ).tobytes()
        probabilities = self.lookup.get(key)
        if probabilities is None:
            self.missing_information_sets.add(key)
            probabilities = np.zeros(self.num_actions, dtype=np.float64)
            probabilities[legal] = 1.0 / float(len(legal))
        return {action: float(probabilities[action]) for action in legal}


def exact_policy_metrics(game, candidate) -> dict:
    tabular = policy.tabular_policy_from_callable(game, candidate.action_probabilities)
    nash_conv = float(exploitability.nash_conv(game, tabular))
    value = float(expected_game_score.policy_value(
        game.new_initial_state(), [tabular] * game.num_players()
    )[0])
    return {
        "nash_conv": nash_conv,
        "exploitability": nash_conv / 2.0,
        "policy_value": value,
        "policy_value_error": abs(value - LEDUC_GAME_VALUE_PLAYER_0),
    }


def exact_neural_policy_metrics(game, model) -> dict:
    return exact_policy_metrics(game, _NeuralPolicy(model))


def exact_empirical_policy_metrics(game, grouped: GroupedReservoir) -> dict:
    empirical = _GroupedPolicy(grouped, grouped.action_probs.shape[1])
    metrics = exact_policy_metrics(game, empirical)
    metrics["missing_information_sets"] = len(empirical.missing_information_sets)
    return metrics


def cross_entropy_objective_equivalence(
    reservoir: FrozenReservoir,
    grouped: GroupedReservoir,
    predictions: np.ndarray,
    *,
    iteration: int,
) -> tuple[float, float]:
    """Return row and grouped CE objectives for a grouped prediction table."""
    lookup = {
        row.tobytes(): prediction
        for row, prediction in zip(grouped.info_states, predictions)
    }
    row_predictions = np.asarray(
        [lookup[row.tobytes()] for row in reservoir.info_states], dtype=np.float64
    )
    row_predictions = np.clip(row_predictions, 1e-12, 1.0)
    row_losses = -np.sum(
        reservoir.action_probs.astype(np.float64) * np.log(row_predictions), axis=1
    )
    row_objective = float(np.mean(
        row_losses * linear_iteration_weights(reservoir, iteration)
    ))
    grouped_predictions = np.clip(np.asarray(predictions, dtype=np.float64), 1e-12, 1.0)
    grouped_losses = -np.sum(
        grouped.action_probs.astype(np.float64) * np.log(grouped_predictions), axis=1
    )
    grouped_objective = float(np.mean(
        grouped_losses * grouped.objective_weights.astype(np.float64)
    ))
    return row_objective, grouped_objective


__all__ = [
    "FrozenReservoir",
    "GroupedReservoir",
    "cross_entropy_objective_equivalence",
    "decode_serialized_reservoir",
    "exact_empirical_policy_metrics",
    "exact_neural_policy_metrics",
    "fit_policy",
    "group_reservoir",
    "initial_policy_weights",
    "load_frozen_reservoir",
    "policy_network_from_weights",
    "save_frozen_reservoir",
    "sha256",
]
