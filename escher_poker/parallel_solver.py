# Adapted from Sandholm-Lab/ESCHER parallelized_ESCHER.py at commit
# e694eaaa251952696aaf36ef1c790887c8324750.
# Original code from OpenSpiel Deep CFR implementation:
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/deep_cfr_tf2.py
# Original Deep CFR code copyright 2019 DeepMind Technologies Limited
# ESCHER code copyright 2022 Stephen McAleer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray-based ESCHER experience collection with a single learner.

The upstream implementation uses one learner and multiple Ray actors that
hold synchronized inference networks and independent replay shards. This
adaptation retains that architecture while using this repository's current
solver, target processing, networks, and diagnostics. Total traversals and
total replay capacity are partitioned across actors rather than multiplied by
the worker count.
"""

from __future__ import annotations

import collections
import os
import time
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

import pyspiel

from .parallel_utils import (
    aggregate_replay_diagnostics,
    partition_total,
    worker_seed,
)
from .seeding import set_seed_tf
from .solver import ESCHERSolver


UPSTREAM_PARALLEL_SOURCE = (
    "https://github.com/Sandholm-Lab/ESCHER/blob/"
    "e694eaaa251952696aaf36ef1c790887c8324750/parallelized_ESCHER.py"
)


class ESCHERExperienceWorker:
    """Ray actor payload for traversal-only ESCHER experience generation."""

    def __init__(
        self,
        game_name: str,
        solver_kwargs: Dict[str, Any],
        worker_seed: int,
    ):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass

        set_seed_tf(int(worker_seed))
        game = pyspiel.load_game(str(game_name))
        self._solver = ESCHERSolver(game, **dict(solver_kwargs))
        if self._solver._use_balanced_probs:  # pylint: disable=protected-access
            self._solver._prepare_fixed_sampling_policy()  # pylint: disable=protected-access

    def ping(self) -> bool:
        return True

    def collect(
        self,
        n: int,
        player: int,
        regret_weights,
        value_weights,
        iteration: int,
        *,
        train_regret: bool,
        train_value: bool,
        record_value: bool,
        track_mean_squares: bool,
        on_policy_prob: float,
        expl: float,
        val_test: bool,
    ) -> Dict[str, int]:
        self._solver.set_weights(regret_weights)
        self._solver.set_val_weights(value_weights)
        self._solver.set_iteration(int(iteration))
        before = self._solver.get_num_nodes()
        self._solver.traverse_game_tree_n_times(
            int(n),
            int(player),
            train_regret=bool(train_regret),
            train_value=bool(train_value),
            record_value=bool(record_value),
            track_mean_squares=bool(track_mean_squares),
            on_policy_prob=float(on_policy_prob),
            expl=float(expl),
            val_test=bool(val_test),
        )
        return {"nodes_touched": int(self._solver.get_num_nodes() - before)}

    def examples(self, player: int):
        info_state = self._solver.get_example_info_state(int(player))
        legal_mask = self._solver.get_example_legal_actions_mask(int(player))
        hist_state = self._solver.get_example_hist_state()
        return {
            "info_state": None if info_state is None else np.asarray(info_state),
            "legal_actions_mask": (
                None if legal_mask is None else np.asarray(legal_mask)
            ),
            "hist_state": None if hist_state is None else np.asarray(hist_state),
        }

    def regret_memories(self, player: int):
        return list(self._solver.get_regret_memories(int(player)))

    def value_memories(self):
        return list(self._solver.get_value_memory())

    def value_test_memories(self):
        return list(self._solver.get_value_memory_test())

    def average_policy_memories(self):
        return list(self._solver.get_average_policy_memories())

    def regret_memory_count(self, player: int) -> int:
        return int(self._solver.get_regret_memory_count(int(player)))

    def average_policy_memory_count(self) -> int:
        return int(self._solver.get_average_policy_memory_count())

    def regret_replay_diagnostics(self, player: int):
        return dict(self._solver.get_regret_replay_diagnostics(int(player)))

    def regret_num_calls(self) -> int:
        return int(self._solver.get_num_calls())

    def clear_value_memories(self) -> None:
        self._solver.clear_val_memories()

    def clear_value_test_memories(self) -> None:
        self._solver.clear_val_memories_test()

    def clear_regret_memories(self) -> None:
        self._solver.clear_regret_buffers()

    def reset_regret_target_consistency(self) -> None:
        self._solver._reset_regret_target_consistency_diagnostics()  # pylint: disable=protected-access

    def regret_target_consistency_stats(self, player: int):
        return dict(
            self._solver._regret_target_consistency[int(player)]  # pylint: disable=protected-access
        )

    def rss_mb(self) -> float:
        return float(self._solver._current_rss_mb())  # pylint: disable=protected-access


class ParallelESCHERSolver(ESCHERSolver):
    """Single-learner ESCHER with Ray-parallel experience collection."""

    def __init__(
        self,
        game,
        *,
        game_name: str,
        parallel_num_workers: int = 3,
        parallel_run_seed: int = 0,
        parallel_ray_address: str | None = None,
        parallel_log_to_driver: bool = False,
        **solver_kwargs,
    ):
        self._parallel_num_workers = int(parallel_num_workers)
        if self._parallel_num_workers < 2:
            raise ValueError("parallel_num_workers must be at least 2.")
        if solver_kwargs.get("save_regret_memories"):
            raise ValueError("Parallel ESCHER does not support disk-backed regret replay.")
        if solver_kwargs.get("save_average_policy_memories"):
            raise ValueError("Parallel ESCHER does not support disk-backed policy replay.")
        if solver_kwargs.get("on_policy_joint_regret_updates", False):
            raise ValueError("Parallel ESCHER currently requires separate player traversals.")
        if solver_kwargs.get("track_sampling_coverage", False):
            raise ValueError("Parallel ESCHER does not yet aggregate coverage counters.")
        if str(solver_kwargs.get("regret_replay_mode", "reservoir")) != "reservoir":
            raise ValueError(
                "Parallel ESCHER currently supports the Experiment 28 reservoir only."
            )

        total_memory_capacity = int(solver_kwargs["memory_capacity"])
        if total_memory_capacity < self._parallel_num_workers:
            raise ValueError(
                "memory_capacity must provide at least one slot per parallel worker."
            )
        super().__init__(game, **solver_kwargs)
        self._game_name = str(game_name)
        self._parallel_run_seed = int(parallel_run_seed)
        self._workers = []
        self._ray = None
        self._owns_ray_runtime = False

        try:
            import ray  # Imported lazily so sequential ESCHER has no Ray startup cost.

            self._ray = ray
            self._owns_ray_runtime = not ray.is_initialized()
            if self._owns_ray_runtime:
                init_kwargs = {
                    "include_dashboard": False,
                    "log_to_driver": bool(parallel_log_to_driver),
                    "ignore_reinit_error": True,
                }
                if parallel_ray_address:
                    init_kwargs["address"] = str(parallel_ray_address)
                else:
                    init_kwargs["num_cpus"] = self._parallel_num_workers
                ray.init(**init_kwargs)

            worker_class = ray.remote(num_cpus=1)(ESCHERExperienceWorker)
            capacities = partition_total(
                total_memory_capacity,
                self._parallel_num_workers,
            )
            for worker_index, capacity in enumerate(capacities):
                worker_kwargs = dict(solver_kwargs)
                worker_kwargs.update({
                    "memory_capacity": int(capacity),
                    "compute_exploitability": False,
                    "save_policy_weights": False,
                    "save_regret_networks": None,
                    "save_regret_memories": None,
                    "save_average_policy_memories": None,
                    "verbose": False,
                })
                seed = worker_seed(self._parallel_run_seed, worker_index)
                self._workers.append(
                    worker_class.remote(
                        self._game_name,
                        worker_kwargs,
                        seed,
                    )
                )
            ray.get([worker.ping.remote() for worker in self._workers])
        except Exception:
            self.close()
            raise

    @property
    def parallel_num_workers(self) -> int:
        return self._parallel_num_workers

    def close(self) -> None:
        ray = getattr(self, "_ray", None)
        if ray is None:
            return
        for worker in getattr(self, "_workers", []):
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self._workers = []
        if self._owns_ray_runtime and ray.is_initialized():
            ray.shutdown()
        self._ray = None

    def _worker_results(self, method_name: str, *args):
        refs = [
            getattr(worker, method_name).remote(*args)
            for worker in self._workers
        ]
        return self._ray.get(refs)

    def traverse_game_tree_n_times(
        self,
        n,
        p,
        train_regret=False,
        train_value=False,
        record_value=False,
        track_mean_squares=True,
        on_policy_prob=0.0,
        expl=0.6,
        val_test=False,
    ):
        traversal_start = time.perf_counter()
        try:
            counts = partition_total(int(n), self._parallel_num_workers)
            regret_weights = self._ray.put(self.get_weights())
            value_weights = self._ray.put(self.get_val_weights())
            refs = []
            first_active = True
            for worker, count in zip(self._workers, counts):
                if count <= 0:
                    continue
                refs.append(worker.collect.remote(
                    int(count),
                    int(p),
                    regret_weights,
                    value_weights,
                    int(self._iteration),
                    train_regret=bool(train_regret),
                    train_value=bool(train_value),
                    record_value=bool(record_value),
                    track_mean_squares=bool(track_mean_squares and first_active),
                    on_policy_prob=float(on_policy_prob),
                    expl=float(expl),
                    val_test=bool(val_test),
                ))
                first_active = False
            results = self._ray.get(refs) if refs else []
            self._nodes_visited += sum(
                int(row["nodes_touched"]) for row in results
            )
        finally:
            elapsed = time.perf_counter() - traversal_start
            if train_regret:
                self._cumulative_regret_traversal_seconds += elapsed
            if train_value:
                self._cumulative_value_traversal_seconds += elapsed

    def init_regret_net(self):
        for player in range(self._num_players):
            self.traverse_game_tree_n_times(
                1,
                player,
                track_mean_squares=False,
            )
            example = self._ray.get(self._workers[0].examples.remote(player))
            if example["info_state"] is None or example["legal_actions_mask"] is None:
                raise RuntimeError(
                    f"Experience worker did not observe a player-{player} infoset."
                )
            self._example_info_state[player] = tf.constant(
                example["info_state"],
                dtype=tf.float32,
            )
            self._example_legal_actions_mask[player] = tf.constant(
                example["legal_actions_mask"],
                dtype=tf.float32,
            )
            self._init_main_regret_network(
                self._example_info_state[player],
                self._example_legal_actions_mask[player],
                player,
            )

    def init_val_net(self):
        example = self._ray.get(self._workers[0].examples.remote(0))
        if example["hist_state"] is None:
            raise RuntimeError("Experience worker did not produce a history state.")
        self._example_hist_state = np.asarray(example["hist_state"])
        legal_mask = example["legal_actions_mask"]
        if legal_mask is None:
            raise RuntimeError("Experience worker did not produce a legal-action mask.")
        self._example_legal_actions_mask[0] = tf.constant(
            legal_mask,
            dtype=tf.float32,
        )
        self._init_main_val_network(
            self._example_hist_state,
            self._example_legal_actions_mask[0],
        )

    def get_regret_memories(self, player):
        data = []
        for shard in self._worker_results("regret_memories", int(player)):
            data.extend(shard)
        return data

    def get_value_memory(self):
        data = []
        for shard in self._worker_results("value_memories"):
            data.extend(shard)
        return data

    def get_value_memory_test(self):
        data = []
        for shard in self._worker_results("value_test_memories"):
            data.extend(shard)
        return data

    def get_average_policy_memories(self):
        data = []
        for shard in self._worker_results("average_policy_memories"):
            data.extend(shard)
        return data

    def get_regret_memory_count(self, player):
        return int(sum(self._worker_results("regret_memory_count", int(player))))

    def get_average_policy_memory_count(self):
        return int(sum(self._worker_results("average_policy_memory_count")))

    def get_regret_replay_diagnostics(self, player):
        return aggregate_replay_diagnostics(
            self._worker_results("regret_replay_diagnostics", int(player))
        )

    def get_num_calls(self):
        return int(sum(self._worker_results("regret_num_calls")))

    def clear_val_memories(self):
        super().clear_val_memories()
        self._worker_results("clear_value_memories")

    def clear_val_memories_test(self):
        super().clear_val_memories_test()
        self._worker_results("clear_value_test_memories")

    def clear_regret_buffers(self):
        super().clear_regret_buffers()
        self._worker_results("clear_regret_memories")

    def _reset_regret_target_consistency_diagnostics(self):
        super()._reset_regret_target_consistency_diagnostics()
        if getattr(self, "_workers", None):
            self._worker_results("reset_regret_target_consistency")

    def _regret_target_consistency_summary(self, player):
        if not getattr(self, "_workers", None):
            return super()._regret_target_consistency_summary(player)
        rows = self._worker_results("regret_target_consistency_stats", int(player))
        combined = collections.Counter()
        for row in rows:
            combined.update(row)
        count = int(combined["count"])
        if count == 0:
            return super()._regret_target_consistency_summary(player)
        return {
            "count": count,
            "bellman_residual_mean": combined["bellman_residual_sum"] / count,
            "bellman_residual_abs_mean": (
                combined["bellman_residual_abs_sum"] / count
            ),
            "bellman_residual_rmse": np.sqrt(
                combined["bellman_residual_sq_sum"] / count
            ),
            "policy_weighted_target_abs_mean": (
                combined["policy_weighted_target_abs_sum"] / count
            ),
            "all_legal_targets_negative_fraction": (
                combined["all_legal_targets_negative_count"] / count
            ),
        }

    def _current_rss_mb(self):
        central = super()._current_rss_mb()
        if not getattr(self, "_workers", None):
            return central
        worker_rss = self._worker_results("rss_mb")
        return float(central + sum(worker_rss))

    def extract_full_model(self):
        raise NotImplementedError(
            "Parallel replay shards are not yet supported by full-model checkpoints."
        )
