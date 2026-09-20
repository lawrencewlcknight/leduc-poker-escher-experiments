"""Training, deferred evaluation and analysis for Experiment 49."""

from __future__ import annotations

import csv
import gc
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/escher_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyspiel

from escher_poker.chart_titles import set_chart_title
from escher_poker.experiment_utils import (
    cleanup_tensorflow_memory,
    json_safe,
    make_escher_solver,
    safe_stats,
)
from escher_poker.seeding import set_seed_tf
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.distillation import (
    decode_serialized_reservoir,
    exact_empirical_policy_metrics,
    exact_neural_policy_metrics,
    fit_policy,
    group_reservoir,
    initial_policy_weights,
    load_frozen_reservoir,
    policy_network_from_weights,
    save_frozen_reservoir,
    sha256,
)

from .config import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    SELECTED_POLICY_ARM,
    validate_config,
)

_LOGGER = logging.getLogger("escher_poker.experiment.paper_36h_trajectory")


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
        handle.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fields})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_config(*, smoke: bool = False) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if smoke:
        config.update(
            {
                "num_iterations": 2,
                "num_traversals": 2,
                "num_val_fn_traversals": 2,
                "training_wall_clock_seconds": 1e-9,
                "checkpoint_interval_seconds": 1.0,
                "target_nodes_touched": 1,
                "training_progress_every": 1,
                "memory_capacity": 128,
                "regret_memory_capacity": 128,
                "value_memory_capacity": 128,
                "value_validation_memory_capacity": 128,
                "average_policy_memory_capacity": 256,
                "policy_network_train_steps": 2,
                "regret_network_train_steps": 1,
                "value_network_train_steps": 1,
                "batch_size_regret": 2,
                "batch_size_value": 2,
                "batch_size_average_policy": 4,
                "policy_network_layers": (8, 8),
                "regret_network_layers": (8, 8),
                "value_network_layers": (8, 8),
                "reservoir_decode_chunk_size": 64,
                "smoke": True,
            }
        )
    else:
        config["smoke"] = False
    validate_config(config, smoke=smoke)
    return config


def _progress_row(solver, seed: int, progress: Mapping) -> dict:
    cumulative_regret = float(progress["cumulative_regret_traversal_seconds"])
    cumulative_value = float(progress["cumulative_value_traversal_seconds"])
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(seed),
        "iteration": int(progress["iteration"]),
        "solver_iteration": int(progress["solver_iteration"]),
        "nodes_touched": int(progress["nodes_touched"]),
        "active_training_seconds": float(progress["wall_clock_seconds"]),
        "training_hours": float(progress["wall_clock_seconds"]) / 3600.0,
        "learning_rate": float(progress["learning_rate"]),
        "value_loss": float(progress["value_loss"]),
        "value_test_loss": float(progress["value_test_loss"]),
        "regret_loss_player_0": float(progress["regret_loss_player_0"]),
        "regret_loss_player_1": float(progress["regret_loss_player_1"]),
        "cumulative_regret_traversal_seconds": cumulative_regret,
        "cumulative_value_traversal_seconds": cumulative_value,
        "cumulative_experience_collection_seconds": cumulative_regret + cumulative_value,
        "average_policy_buffer_size": int(solver.get_average_policy_memory_count()),
        "regret_buffer_size_player_0": int(solver.get_regret_memory_count(0)),
        "regret_buffer_size_player_1": int(solver.get_regret_memory_count(1)),
        "value_buffer_size": len(solver.get_value_memory()),
        "value_test_buffer_size": len(solver.get_value_memory_test()),
        "peak_rss_mb": float(solver._current_rss_mb()),  # pylint: disable=protected-access
    }


def _freeze_checkpoint(
    *,
    solver,
    seed_dir: Path,
    run_dir: Path,
    config: Mapping,
    checkpoint_index: int,
    progress: Mapping,
    time_checkpoint_index: int | None,
    scheduled_active_seconds: float | None,
    crossed_scheduled_count: int,
    is_node_15m_endpoint: bool,
    is_final_endpoint: bool,
) -> dict:
    checkpoint_id = f"checkpoint_{checkpoint_index:03d}"
    serialized = list(solver.get_average_policy_memories())
    if not serialized:
        raise RuntimeError("Cannot freeze an empty average-policy reservoir")
    frozen = decode_serialized_reservoir(
        serialized,
        solver._average_policy_feature_description,  # pylint: disable=protected-access
        chunk_size=int(config["reservoir_decode_chunk_size"]),
    )
    path = seed_dir / str(config["checkpoint_directory"]) / f"{checkpoint_id}.npz"
    manifest = save_frozen_reservoir(path, frozen)
    row = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(progress.get("seed", 0)),
        "checkpoint_id": checkpoint_id,
        "checkpoint_index": int(checkpoint_index),
        "time_checkpoint_index": time_checkpoint_index,
        "scheduled_active_seconds": scheduled_active_seconds,
        "crossed_scheduled_count": int(crossed_scheduled_count),
        "iteration": int(progress["iteration"]),
        "solver_iteration": int(progress["solver_iteration"]),
        "nodes_touched": int(progress["nodes_touched"]),
        "active_training_seconds": float(progress["wall_clock_seconds"]),
        "training_hours": float(progress["wall_clock_seconds"]) / 3600.0,
        "is_node_15m_endpoint": bool(is_node_15m_endpoint),
        "is_final_endpoint": bool(is_final_endpoint),
        "reservoir_path": str(path.relative_to(run_dir)),
        "reservoir_sha256": manifest["sha256"],
        "reservoir_size_bytes": manifest["size_bytes"],
        "reservoir_rows": manifest["num_rows"],
        "info_state_width": manifest["info_state_width"],
        "num_actions": manifest["num_actions"],
    }
    del serialized, frozen
    gc.collect()
    return row


def run_training_seed(
    *,
    seed: int,
    output_dir: Path,
    smoke: bool = False,
) -> dict:
    """Train one source learner and freeze time/node/final reservoirs."""
    config = build_config(smoke=smoke)
    output_dir = Path(output_dir)
    seed_dir = output_dir / f"seed_{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "config.json", config)
    set_seed_tf(int(seed))
    game = pyspiel.load_game(str(config["game_name"]))
    solver = make_escher_solver(game, config, run_seed=int(seed))

    checkpoint_rows: list[dict] = []
    progress_rows: list[dict] = []
    latest_progress: dict = {}
    interval = float(config["checkpoint_interval_seconds"])
    next_scheduled = interval
    next_time_index = 1
    node_endpoint_saved = False

    def persist_manifest() -> None:
        write_csv(seed_dir / str(config["checkpoint_manifest_filename"]), checkpoint_rows)

    def record_checkpoint(
        current_solver,
        progress: Mapping,
        *,
        time_index: int | None,
        scheduled_seconds: float | None,
        crossed_count: int,
        node_endpoint: bool,
        final_endpoint: bool,
    ) -> dict:
        if checkpoint_rows and int(checkpoint_rows[-1]["solver_iteration"]) == int(
            progress["solver_iteration"]
        ):
            row = checkpoint_rows[-1]
            if time_index is not None:
                row["time_checkpoint_index"] = int(time_index)
                row["scheduled_active_seconds"] = float(scheduled_seconds)
                row["crossed_scheduled_count"] = int(crossed_count)
            row["is_node_15m_endpoint"] = bool(row["is_node_15m_endpoint"] or node_endpoint)
            row["is_final_endpoint"] = bool(row["is_final_endpoint"] or final_endpoint)
            persist_manifest()
            return row
        enriched = dict(progress, seed=int(seed))
        row = _freeze_checkpoint(
            solver=current_solver,
            seed_dir=seed_dir,
            run_dir=output_dir,
            config=config,
            checkpoint_index=len(checkpoint_rows),
            progress=enriched,
            time_checkpoint_index=time_index,
            scheduled_active_seconds=scheduled_seconds,
            crossed_scheduled_count=crossed_count,
            is_node_15m_endpoint=node_endpoint,
            is_final_endpoint=final_endpoint,
        )
        checkpoint_rows.append(row)
        persist_manifest()
        _LOGGER.info(
            "Seed %s froze %s at %.3f active hours, %s nodes",
            seed,
            row["checkpoint_id"],
            row["training_hours"],
            row["nodes_touched"],
        )
        return row

    def post_iteration(current_solver, progress: Mapping) -> None:
        nonlocal next_scheduled, next_time_index, node_endpoint_saved
        latest_progress.clear()
        latest_progress.update(progress)
        solver_iteration = int(progress["solver_iteration"])
        if solver_iteration == 1 or solver_iteration % int(config["training_progress_every"]) == 0:
            progress_rows.append(_progress_row(current_solver, int(seed), progress))

        active_seconds = float(progress["wall_clock_seconds"])
        time_index = None
        scheduled_seconds = None
        crossed_count = 0
        if active_seconds >= next_scheduled:
            time_index = next_time_index
            scheduled_seconds = next_scheduled
            while active_seconds >= next_scheduled:
                crossed_count += 1
                next_scheduled += interval
                next_time_index += 1
        node_endpoint = not node_endpoint_saved and int(progress["nodes_touched"]) >= int(
            config["target_nodes_touched"]
        )
        if node_endpoint:
            node_endpoint_saved = True
        if time_index is not None or node_endpoint:
            record_checkpoint(
                current_solver,
                progress,
                time_index=time_index,
                scheduled_seconds=scheduled_seconds,
                crossed_count=crossed_count,
                node_endpoint=node_endpoint,
                final_endpoint=False,
            )

    outer_started = time.perf_counter()
    _, policy_loss, convs, nodes, values, _ = solver.solve(
        max_wall_clock_seconds=float(config["training_wall_clock_seconds"]),
        post_iteration_callback=post_iteration,
        exclude_post_iteration_callback_time=True,
        fit_final_policy=False,
    )
    outer_seconds = time.perf_counter() - outer_started
    summary = solver.get_last_solve_summary()
    if summary is None or not latest_progress:
        raise RuntimeError("The source learner did not publish completion metadata")
    if convs or nodes or values:
        raise RuntimeError("Policy evaluation occurred inside the timed learner")
    if not math.isnan(float(np.asarray(policy_loss))):
        raise RuntimeError("The source learner unexpectedly fitted a final policy")
    if not smoke and not bool(summary["hit_wall_clock_limit"]):
        raise RuntimeError("Training stopped before the 36-hour active-time boundary")

    record_checkpoint(
        solver,
        latest_progress,
        time_index=None,
        scheduled_seconds=None,
        crossed_count=0,
        node_endpoint=(
            not node_endpoint_saved
            and int(latest_progress["nodes_touched"]) >= int(config["target_nodes_touched"])
        ),
        final_endpoint=True,
    )
    if progress_rows and int(progress_rows[-1]["solver_iteration"]) != int(
        latest_progress["solver_iteration"]
    ):
        progress_rows.append(_progress_row(solver, int(seed), latest_progress))
    write_csv(seed_dir / "source_training_progress.csv", progress_rows)

    result = {
        "status": "complete",
        "stage": "training",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(seed),
        "smoke": bool(smoke),
        "active_training_seconds": float(summary["active_training_seconds"]),
        "training_budget_seconds": float(config["training_wall_clock_seconds"]),
        "budget_overshoot_seconds": float(summary["budget_overshoot_seconds"]),
        "checkpoint_time_excluded_seconds": float(
            summary["excluded_post_iteration_callback_seconds"]
        ),
        "outer_training_stage_seconds": float(outer_seconds),
        "completed_solve_passes": int(summary["completed_solve_passes"]),
        "final_solver_iteration": int(summary["solver_iteration"]),
        "final_nodes_touched": int(summary["nodes_touched"]),
        "hit_wall_clock_limit": bool(summary["hit_wall_clock_limit"]),
        "num_checkpoints": len(checkpoint_rows),
        "num_time_checkpoints": sum(
            row["time_checkpoint_index"] is not None for row in checkpoint_rows
        ),
        "node_15m_endpoint_saved": any(row["is_node_15m_endpoint"] for row in checkpoint_rows),
        "manifest_path": str(
            (seed_dir / str(config["checkpoint_manifest_filename"])).relative_to(output_dir)
        ),
        "progress_path": str((seed_dir / "source_training_progress.csv").relative_to(output_dir)),
    }
    write_json(seed_dir / "training_result.json", result)
    write_json(seed_dir / "TRAINING_SUCCESS.json", {"status": "complete", "seed": int(seed)})
    del solver
    cleanup_tensorflow_memory()
    gc.collect()
    return result


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _optional_int(value) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _checkpoint_metric_is_valid(
    path: Path,
    *,
    seed: int,
    reservoir_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        row.get("status") == "complete"
        and int(row.get("experiment_id", -1)) == EXPERIMENT_ID
        and int(row.get("seed", -1)) == int(seed)
        and row.get("reservoir_sha256") == reservoir_sha256
        and row.get("policy_arm") == SELECTED_POLICY_ARM
    )


def run_evaluation_seed(
    *,
    seed: int,
    training_dir: Path,
    output_dir: Path,
    smoke: bool = False,
    resume: bool = True,
) -> dict:
    """Fit and evaluate the selected policy independently at every snapshot."""
    config = build_config(smoke=smoke)
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    source_seed_dir = training_dir / f"seed_{int(seed)}"
    evaluation_seed_dir = output_dir / f"seed_{int(seed)}"
    evaluation_seed_dir.mkdir(parents=True, exist_ok=True)
    training_result = read_json(source_seed_dir / "training_result.json")
    if training_result.get("status") != "complete":
        raise RuntimeError(f"Seed {seed} has no complete training result")
    manifest_path = training_dir / str(training_result["manifest_path"])
    source_progress_path = training_dir / str(training_result["progress_path"])
    copied_progress_path = evaluation_seed_dir / "source_training_progress.csv"
    shutil.copy2(source_progress_path, copied_progress_path)
    manifest_rows = read_csv(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"Seed {seed} has an empty checkpoint manifest")
    for row in manifest_rows:
        reservoir_path = training_dir / str(row["reservoir_path"])
        if not reservoir_path.is_file() or sha256(reservoir_path) != row["reservoir_sha256"]:
            raise RuntimeError(f"Checkpoint reservoir failed validation: {reservoir_path}")

    game = pyspiel.load_game(str(config["game_name"]))
    first_reservoir = load_frozen_reservoir(training_dir / str(manifest_rows[0]["reservoir_path"]))
    fit_seed = int(config["fit_seed_offset"]) + int(seed)
    base_weights = initial_policy_weights(config, first_reservoir, fit_seed=fit_seed)
    initial_model = policy_network_from_weights(config, first_reservoir, base_weights)
    initial_metrics = exact_neural_policy_metrics(game, initial_model)
    del initial_model, first_reservoir
    cleanup_tensorflow_memory()

    metric_rows: list[dict] = [
        {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "seed": int(seed),
            "checkpoint_id": "initial_policy",
            "checkpoint_index": -1,
            "time_checkpoint_index": 0,
            "scheduled_active_seconds": 0.0,
            "iteration": 0,
            "solver_iteration": 0,
            "nodes_touched": 0,
            "active_training_seconds": 0.0,
            "training_hours": 0.0,
            "is_initial_policy": True,
            "is_node_15m_endpoint": False,
            "is_final_endpoint": False,
            "policy_arm": SELECTED_POLICY_ARM,
            "fit_seed": fit_seed,
            "neural_exploitability": initial_metrics["exploitability"],
            "neural_nash_conv": initial_metrics["nash_conv"],
            "neural_policy_value": initial_metrics["policy_value"],
            "neural_policy_value_error": initial_metrics["policy_value_error"],
            "empirical_reservoir_exploitability": np.nan,
            "empirical_reservoir_policy_value": np.nan,
            "distillation_gap": np.nan,
            "reservoir_rows": 0,
            "unique_information_sets": 0,
            "reservoir_sha256": "",
            "fit_seconds": 0.0,
            "optimizer_steps": 0,
            "network_examples_processed": 0,
            "target_network_examples": 0,
            "final_training_loss": np.nan,
            "weights_path": "",
        }
    ]

    checkpoint_metrics_dir = evaluation_seed_dir / "checkpoint_metrics"
    checkpoint_metrics_dir.mkdir(parents=True, exist_ok=True)
    for position, source in enumerate(manifest_rows, start=1):
        checkpoint_id = str(source["checkpoint_id"])
        metric_path = checkpoint_metrics_dir / f"{checkpoint_id}.json"
        reservoir_path = training_dir / str(source["reservoir_path"])
        if resume and _checkpoint_metric_is_valid(
            metric_path,
            seed=int(seed),
            reservoir_sha256=str(source["reservoir_sha256"]),
        ):
            row = read_json(metric_path)
            metric_rows.append(row)
            _LOGGER.info(
                "Seed %s reused evaluation %s (%s/%s)",
                seed,
                checkpoint_id,
                position,
                len(manifest_rows),
            )
            continue

        frozen = load_frozen_reservoir(reservoir_path)
        source_iteration = int(source["solver_iteration"])
        grouped = group_reservoir(frozen, iteration=source_iteration)
        empirical = exact_empirical_policy_metrics(game, grouped)
        fit_config = dict(config, source_final_iteration=source_iteration)
        model, fit_diagnostics = fit_policy(
            frozen,
            grouped,
            fit_config,
            loss_name=str(config["selected_policy_loss"]),
            use_grouped_data=bool(config["selected_policy_grouped"]),
            example_multiplier=int(config["selected_policy_example_multiplier"]),
            fit_seed=fit_seed,
            base_weights=base_weights,
        )
        neural = exact_neural_policy_metrics(game, model)
        endpoint = _bool(source["is_node_15m_endpoint"]) or _bool(source["is_final_endpoint"])
        weights_path = ""
        if endpoint:
            absolute_weights = evaluation_seed_dir / f"{checkpoint_id}.weights.h5"
            model.save_weights(str(absolute_weights))
            weights_path = str(absolute_weights.relative_to(output_dir))
        row = {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "seed": int(seed),
            "checkpoint_id": checkpoint_id,
            "checkpoint_index": int(source["checkpoint_index"]),
            "time_checkpoint_index": _optional_int(source["time_checkpoint_index"]),
            "scheduled_active_seconds": _optional_float(source["scheduled_active_seconds"]),
            "iteration": int(source["iteration"]),
            "solver_iteration": source_iteration,
            "nodes_touched": int(source["nodes_touched"]),
            "active_training_seconds": float(source["active_training_seconds"]),
            "training_hours": float(source["training_hours"]),
            "is_initial_policy": False,
            "is_node_15m_endpoint": _bool(source["is_node_15m_endpoint"]),
            "is_final_endpoint": _bool(source["is_final_endpoint"]),
            "policy_arm": SELECTED_POLICY_ARM,
            "fit_seed": fit_seed,
            "neural_exploitability": neural["exploitability"],
            "neural_nash_conv": neural["nash_conv"],
            "neural_policy_value": neural["policy_value"],
            "neural_policy_value_error": neural["policy_value_error"],
            "empirical_reservoir_exploitability": empirical["exploitability"],
            "empirical_reservoir_policy_value": empirical["policy_value"],
            "empirical_missing_information_sets": empirical["missing_information_sets"],
            "distillation_gap": (neural["exploitability"] - empirical["exploitability"]),
            "reservoir_rows": frozen.size,
            "unique_information_sets": grouped.size,
            "reservoir_sha256": str(source["reservoir_sha256"]),
            "weights_path": weights_path,
            **fit_diagnostics,
        }
        write_json(metric_path, row)
        metric_rows.append(row)
        write_csv(evaluation_seed_dir / str(config["checkpoint_metrics_filename"]), metric_rows)
        _LOGGER.info(
            "Seed %s evaluated %s (%s/%s): exploitability %.6f",
            seed,
            checkpoint_id,
            position,
            len(manifest_rows),
            neural["exploitability"],
        )
        del model, grouped, frozen
        cleanup_tensorflow_memory()
        gc.collect()

    write_csv(evaluation_seed_dir / str(config["checkpoint_metrics_filename"]), metric_rows)
    result = {
        "status": "complete",
        "stage": "evaluation",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(seed),
        "smoke": bool(smoke),
        "policy_arm": SELECTED_POLICY_ARM,
        "num_source_checkpoints": len(manifest_rows),
        "num_metric_rows": len(metric_rows),
        "num_node_15m_endpoints": sum(
            _bool(row.get("is_node_15m_endpoint")) for row in metric_rows
        ),
        "num_final_endpoints": sum(_bool(row.get("is_final_endpoint")) for row in metric_rows),
        "metrics_path": str(
            (evaluation_seed_dir / str(config["checkpoint_metrics_filename"])).relative_to(
                output_dir
            )
        ),
        "progress_path": str(copied_progress_path.relative_to(output_dir)),
    }
    write_json(evaluation_seed_dir / "evaluation_result.json", result)
    write_json(
        evaluation_seed_dir / "EVALUATION_SUCCESS.json",
        {"status": "complete", "seed": int(seed)},
    )
    return result


def _mean_summary(rows: Sequence[Mapping]) -> list[dict]:
    summary = []
    indices = sorted(
        {
            int(row["time_checkpoint_index"])
            for row in rows
            if row.get("time_checkpoint_index") not in (None, "", "None")
        }
    )
    for index in indices:
        selected = [
            row
            for row in rows
            if row.get("time_checkpoint_index") not in (None, "", "None")
            and int(row["time_checkpoint_index"]) == index
        ]
        neural = safe_stats([float(row["neural_exploitability"]) for row in selected])
        empirical_values = [
            float(row["empirical_reservoir_exploitability"])
            for row in selected
            if row.get("empirical_reservoir_exploitability") not in (None, "", "None")
        ]
        empirical = safe_stats(empirical_values)
        gaps = [
            float(row["distillation_gap"])
            for row in selected
            if row.get("distillation_gap") not in (None, "", "None")
        ]
        gap = safe_stats(gaps)
        nodes = safe_stats([float(row["nodes_touched"]) for row in selected])
        hours = safe_stats([float(row["training_hours"]) for row in selected])
        summary.append(
            {
                "time_checkpoint_index": index,
                "mean_training_hours": hours["mean"],
                "std_training_hours": hours["std"],
                "mean_nodes_touched": nodes["mean"],
                "std_nodes_touched": nodes["std"],
                "mean_neural_exploitability": neural["mean"],
                "std_neural_exploitability": neural["std"],
                "se_neural_exploitability": neural["se"],
                "mean_empirical_exploitability": empirical["mean"],
                "std_empirical_exploitability": empirical["std"],
                "se_empirical_exploitability": empirical["se"],
                "mean_distillation_gap": gap["mean"],
                "std_distillation_gap": gap["std"],
                "se_distillation_gap": gap["se"],
                "n_seeds": neural["n_finite"],
                "n_empirical_seeds": empirical["n_finite"],
            }
        )
    return summary


def _node_aligned_summary(
    rows: Sequence[Mapping], *, target_nodes: int, grid_points: int = 73
) -> list[dict]:
    """Interpolate seed trajectories onto a common touched-node grid."""
    seeds = sorted({int(row["seed"]) for row in rows})
    paths = {}
    for seed in seeds:
        selected = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: float(row["nodes_touched"]),
        )
        paths[seed] = selected
    common_max = min(max(float(row["nodes_touched"]) for row in path) for path in paths.values())
    grid = np.linspace(0.0, common_max, int(grid_points))
    if 0 < int(target_nodes) <= common_max:
        grid = np.unique(np.concatenate((grid, [float(target_nodes)])))
    summary = []
    for node in grid:
        neural_values, empirical_values, gap_values = [], [], []
        for path in paths.values():
            x = np.asarray([float(row["nodes_touched"]) for row in path])
            neural = np.asarray([float(row["neural_exploitability"]) for row in path])
            neural_values.append(float(np.interp(node, x, neural)))
            empirical_path = [
                row
                for row in path
                if row.get("empirical_reservoir_exploitability") not in (None, "", "None")
            ]
            empirical_x = np.asarray([float(row["nodes_touched"]) for row in empirical_path])
            if empirical_x.size and node >= empirical_x[0]:
                empirical_y = np.asarray(
                    [float(row["empirical_reservoir_exploitability"]) for row in empirical_path]
                )
                empirical_value = float(np.interp(node, empirical_x, empirical_y))
                empirical_values.append(empirical_value)
                gap_values.append(neural_values[-1] - empirical_value)
        neural_stats = safe_stats(neural_values)
        empirical_stats = safe_stats(empirical_values)
        gap_stats = safe_stats(gap_values)
        summary.append(
            {
                "nodes_touched": float(node),
                "mean_neural_exploitability": neural_stats["mean"],
                "std_neural_exploitability": neural_stats["std"],
                "se_neural_exploitability": neural_stats["se"],
                "mean_empirical_exploitability": empirical_stats["mean"],
                "std_empirical_exploitability": empirical_stats["std"],
                "se_empirical_exploitability": empirical_stats["se"],
                "mean_distillation_gap": gap_stats["mean"],
                "std_distillation_gap": gap_stats["std"],
                "se_distillation_gap": gap_stats["se"],
                "n_seeds": neural_stats["n_finite"],
                "n_empirical_seeds": empirical_stats["n_finite"],
            }
        )
    return summary


def _plot_exploitability(
    rows: Sequence[Mapping],
    summary: Sequence[Mapping],
    *,
    x_field: str,
    summary_x_field: str,
    xlabel: str,
    output: Path,
    include_endpoint_points: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = sorted(
            (
                row
                for row in rows
                if int(row["seed"]) == seed
                and (
                    include_endpoint_points
                    or row.get("time_checkpoint_index") not in (None, "", "None")
                )
            ),
            key=lambda row: float(row[x_field]),
        )
        ax.plot(
            [float(row[x_field]) for row in selected],
            [float(row["neural_exploitability"]) for row in selected],
            color="#2878B5",
            alpha=0.20,
            linewidth=1.0,
        )
    x = np.asarray([float(row[summary_x_field]) for row in summary])
    neural = np.asarray([float(row["mean_neural_exploitability"]) for row in summary])
    neural_se = np.asarray([float(row["se_neural_exploitability"]) for row in summary])
    empirical = np.asarray([float(row["mean_empirical_exploitability"]) for row in summary])
    empirical_se = np.asarray([float(row["se_empirical_exploitability"]) for row in summary])
    ax.plot(x, neural, color="#174A73", linewidth=2.2, label="Neural average policy")
    ax.fill_between(
        x,
        neural - neural_se,
        neural + neural_se,
        color="#2878B5",
        alpha=0.20,
        label="Neural ±1 SE",
    )
    finite = np.isfinite(empirical)
    ax.plot(
        x[finite],
        empirical[finite],
        color="#E45756",
        linewidth=2.0,
        linestyle="--",
        label="Empirical reservoir policy",
    )
    ax.fill_between(
        x[finite],
        (empirical - empirical_se)[finite],
        (empirical + empirical_se)[finite],
        color="#E45756",
        alpha=0.12,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 49: paper-aligned ESCHER convergence")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_gap(summary: Sequence[Mapping], output: Path) -> None:
    selected = [row for row in summary if int(row["time_checkpoint_index"]) > 0]
    x = np.asarray([float(row["mean_training_hours"]) for row in selected])
    mean = np.asarray([float(row["mean_distillation_gap"]) for row in selected])
    se = np.asarray([float(row["se_distillation_gap"]) for row in selected])
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.plot(x, mean, color="#6F4E7C", linewidth=2.2)
    ax.fill_between(x, mean - se, mean + se, color="#6F4E7C", alpha=0.20)
    ax.set_xlabel("Active source-training time (hours)")
    ax.set_ylabel("Neural minus empirical exploitability")
    set_chart_title(ax, "Experiment 49: average-policy distillation gap")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _endpoint_summary(
    rows: Sequence[Mapping], flag: str, endpoint_id: str
) -> tuple[list[dict], list[dict]]:
    selected = [row for row in rows if _bool(row.get(flag))]
    by_seed = []
    for row in sorted(selected, key=lambda item: int(item["seed"])):
        by_seed.append(
            {
                "endpoint_id": endpoint_id,
                "seed": int(row["seed"]),
                "checkpoint_id": row["checkpoint_id"],
                "solver_iteration": int(row["solver_iteration"]),
                "nodes_touched": int(row["nodes_touched"]),
                "active_training_seconds": float(row["active_training_seconds"]),
                "training_hours": float(row["training_hours"]),
                "neural_exploitability": float(row["neural_exploitability"]),
                "empirical_reservoir_exploitability": float(
                    row["empirical_reservoir_exploitability"]
                ),
                "distillation_gap": float(row["distillation_gap"]),
                "weights_path": row.get("weights_path", ""),
            }
        )
    aggregate = []
    for metric in (
        "nodes_touched",
        "training_hours",
        "neural_exploitability",
        "empirical_reservoir_exploitability",
        "distillation_gap",
    ):
        stats = safe_stats([float(row[metric]) for row in by_seed])
        aggregate.append({"endpoint_id": endpoint_id, "metric": metric, **stats})
    return by_seed, aggregate


def _thesis_seed_summary(rows: Sequence[Mapping], target_nodes: int) -> list[dict]:
    result = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = sorted(
            (
                row
                for row in rows
                if int(row["seed"]) == seed
                and row.get("neural_exploitability") not in (None, "", "None")
            ),
            key=lambda row: float(row["nodes_touched"]),
        )
        node_endpoint = next(row for row in seed_rows if _bool(row.get("is_node_15m_endpoint")))
        x = np.asarray([float(row["nodes_touched"]) for row in seed_rows])
        y = np.asarray([float(row["neural_exploitability"]) for row in seed_rows])
        keep = x < float(target_nodes)
        auc_x = np.concatenate((x[keep], [float(target_nodes)]))
        auc_y = np.concatenate((y[keep], [float(np.interp(target_nodes, x, y))]))
        late = [
            float(row["neural_exploitability"])
            for row in seed_rows
            if target_nodes - 1_000_000
            <= float(row["nodes_touched"])
            <= float(node_endpoint["nodes_touched"])
        ]
        result.append(
            {
                "seed": seed,
                "endpoint_nodes_touched": int(node_endpoint["nodes_touched"]),
                "endpoint_training_hours": float(node_endpoint["training_hours"]),
                "endpoint_neural_exploitability": float(node_endpoint["neural_exploitability"]),
                "endpoint_empirical_exploitability": float(
                    node_endpoint["empirical_reservoir_exploitability"]
                ),
                "endpoint_distillation_gap": float(node_endpoint["distillation_gap"]),
                "late_1m_neural_exploitability_mean": float(np.mean(late)),
                "normalised_auc_0_to_15m": float(np.trapz(auc_y, auc_x) / float(target_nodes)),
            }
        )
    return result


def aggregate_results(
    *,
    metric_rows: Sequence[Mapping],
    training_rows: Sequence[Mapping],
    output_dir: Path,
    expected_seed_count: int,
    target_nodes: int,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(row["seed"]) for row in metric_rows})
    if len(seeds) != int(expected_seed_count):
        raise ValueError(f"Expected {expected_seed_count} evaluated seeds, found {len(seeds)}")
    write_csv(output_dir / "checkpoint_metrics.csv", metric_rows)
    write_csv(output_dir / "source_training_progress.csv", training_rows)
    summary = _mean_summary(metric_rows)
    write_csv(output_dir / "trajectory_summary.csv", summary)
    node_summary = _node_aligned_summary(metric_rows, target_nodes=int(target_nodes))
    write_csv(output_dir / "trajectory_by_nodes_summary.csv", node_summary)
    _plot_exploitability(
        metric_rows,
        summary,
        x_field="training_hours",
        summary_x_field="mean_training_hours",
        xlabel="Active source-training time (hours)",
        output=output_dir / "exploitability_by_training_time.png",
    )
    _plot_exploitability(
        metric_rows,
        node_summary,
        x_field="nodes_touched",
        summary_x_field="nodes_touched",
        xlabel="Nodes touched",
        output=output_dir / "exploitability_by_nodes.png",
        include_endpoint_points=True,
    )
    _plot_gap(summary, output_dir / "distillation_gap_by_training_time.png")

    node_rows, node_summary = _endpoint_summary(metric_rows, "is_node_15m_endpoint", "node_15m")
    final_rows, final_summary = _endpoint_summary(metric_rows, "is_final_endpoint", "time_36h")
    if len(node_rows) != len(seeds):
        raise ValueError("Not every seed produced a 15-million-node endpoint")
    if len(final_rows) != len(seeds):
        raise ValueError("Not every seed produced a 36-hour endpoint")
    write_csv(output_dir / "endpoint_15m_seed_metrics.csv", node_rows)
    write_csv(output_dir / "endpoint_36h_seed_metrics.csv", final_rows)
    write_csv(output_dir / "endpoint_summary.csv", node_summary + final_summary)

    thesis_rows = _thesis_seed_summary(metric_rows, int(target_nodes))
    write_csv(output_dir / "thesis_15m_seed_summary.csv", thesis_rows)
    thesis_aggregate = []
    for metric in (
        "endpoint_neural_exploitability",
        "endpoint_empirical_exploitability",
        "endpoint_distillation_gap",
        "late_1m_neural_exploitability_mean",
        "normalised_auc_0_to_15m",
    ):
        stats = safe_stats([float(row[metric]) for row in thesis_rows])
        thesis_aggregate.append({"metric": metric, **stats})
    write_csv(output_dir / "thesis_15m_aggregate_summary.csv", thesis_aggregate)
    result = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_metric_rows": len(metric_rows),
        "num_training_progress_rows": len(training_rows),
        "num_regular_time_points": len(summary),
        "policy_arm": SELECTED_POLICY_ARM,
        "target_nodes_touched": int(target_nodes),
        "endpoint_summary": node_summary + final_summary,
        "thesis_15m_summary": thesis_aggregate,
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = [
    "aggregate_results",
    "build_config",
    "read_csv",
    "read_json",
    "run_evaluation_seed",
    "run_training_seed",
    "write_csv",
    "write_json",
]
