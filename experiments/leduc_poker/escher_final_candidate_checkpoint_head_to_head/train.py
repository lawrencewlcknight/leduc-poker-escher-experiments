"""Uninterrupted ESCHER training with lightweight temporal snapshots."""

from __future__ import annotations

import csv
import gc
import logging
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Sequence

import pyspiel
import tensorflow as tf

from escher_poker.experiment_utils import make_escher_solver
from escher_poker.policy_snapshots import policy_snapshot_path, save_policy_snapshot
from escher_poker.seeding import set_seed_tf

from .config import validate_config


_LOGGER = logging.getLogger(__name__)


def _write_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cleanup_solver(solver) -> None:
    if solver is not None:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _run_seed(
    *,
    seed: int,
    config: Mapping[str, object],
    run_dir: Path,
    existing_rows: List[dict],
    metrics_path: Path,
) -> List[dict]:
    seed_config = deepcopy(dict(config))
    set_seed_tf(seed)
    game = pyspiel.load_game(str(seed_config["game_name"]))
    solver = make_escher_solver(game, seed_config, run_seed=seed)
    schedule = {int(value) for value in seed_config["checkpoint_schedule"]}
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    training_start = time.perf_counter()
    seed_rows: List[dict] = []

    def save_requested_snapshot(active_solver, evaluation_iteration: int) -> None:
        if evaluation_iteration not in schedule:
            return
        path = policy_snapshot_path(
            snapshots_dir,
            seed,
            evaluation_iteration,
            "checkpointed",
        )
        stage_index = tuple(seed_config["checkpoint_schedule"]).index(
            evaluation_iteration
        ) + 1
        save_policy_snapshot(
            active_solver,
            path,
            seed=seed,
            iteration=evaluation_iteration,
            arm="checkpointed",
            config=seed_config,
            stage_label=f"straight-through checkpoint {evaluation_iteration}",
        )
        row = {
            "seed": int(seed),
            "checkpoint_iteration": int(evaluation_iteration),
            "checkpoint_stage": int(stage_index),
            "checkpoint_fraction": float(
                evaluation_iteration / int(seed_config["num_iterations"])
            ),
            "completed_solve_passes": int(getattr(active_solver, "_iteration", -1)),
            "solve_pass_fraction": float(
                int(getattr(active_solver, "_iteration", -1))
                / (int(seed_config["num_iterations"]) + 1)
            ),
            "nodes_touched": int(active_solver.get_num_nodes()),
            "wall_clock_seconds": float(time.perf_counter() - training_start),
            "average_policy_buffer_size": int(
                active_solver.get_average_policy_memory_count()
            ),
            "regret_buffer_size_player_0": int(
                active_solver.get_regret_memory_count(0)
            ),
            "regret_buffer_size_player_1": int(
                active_solver.get_regret_memory_count(1)
            ),
            "policy_snapshot": str(path.resolve()),
        }
        seed_rows.append(row)
        _write_rows(metrics_path, [*existing_rows, *seed_rows])
        _LOGGER.info(
            "Saved seed %s checkpoint %s (pass %s) at %s nodes",
            seed,
            evaluation_iteration,
            row["completed_solve_passes"],
            row["nodes_touched"],
        )

    try:
        solver.solve(post_evaluation_callback=save_requested_snapshot)
        captured = {int(row["checkpoint_iteration"]) for row in seed_rows}
        missing = sorted(schedule - captured)
        if missing:
            raise RuntimeError(f"Training completed without snapshots at {missing}")
        return seed_rows
    finally:
        _cleanup_solver(solver)


def run_training(
    *,
    config: Mapping[str, object],
    seeds: Sequence[int],
    run_dir: Path,
    continue_on_error: bool = True,
) -> dict:
    """Train one uninterrupted Experiment 28 trajectory for each seed."""
    validate_config(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "training_stage_metrics.csv"
    metrics_rows: List[dict] = []
    failed: List[dict] = []

    for seed_value in seeds:
        seed = int(seed_value)
        _LOGGER.info("Starting straight-through ESCHER checkpoint run for seed %s", seed)
        try:
            metrics_rows.extend(
                _run_seed(
                    seed=seed,
                    config=config,
                    run_dir=run_dir,
                    existing_rows=metrics_rows,
                    metrics_path=metrics_path,
                )
            )
            _write_rows(metrics_path, metrics_rows)
        except Exception as exc:  # pragma: no cover - cloud failure path
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failed.append({
                "seed": seed,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            if not continue_on_error:
                break

    return {
        "metrics_rows": metrics_rows,
        "failed": failed,
        "metrics_csv": metrics_path,
        "snapshots_dir": run_dir / "snapshots",
    }


__all__ = ["run_training", "validate_config"]
