"""Run Experiment 45: standard-ESCHER frozen policy distillation audit."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Iterable, Mapping, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/escher_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
import tensorflow as tf  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.experiment_utils import (  # noqa: E402
    cleanup_tensorflow_memory,
    create_run_dir,
    json_safe,
    make_escher_solver,
    safe_stats,
)
from escher_poker.seeding import set_seed_tf  # noqa: E402

from .config import (  # noqa: E402
    ARMS,
    ARM_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    validate_config,
)
from .distillation import (  # noqa: E402
    decode_serialized_reservoir,
    exact_empirical_policy_metrics,
    exact_neural_policy_metrics,
    fit_policy,
    group_reservoir,
    initial_policy_weights,
    load_frozen_reservoir,
    save_frozen_reservoir,
)


_LOGGER = logging.getLogger("escher_poker.experiment.frozen_policy_distillation")


def _parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_seeds(value: Optional[str], *, smoke: bool) -> list[int]:
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [1234] if smoke else list(DEFAULT_SEEDS)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        writer.writerows([{key: json_safe(row.get(key)) for key in fields} for row in rows])


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "check_exploitability_every": args.evaluation_interval,
        "memory_capacity": args.memory_capacity,
        "average_policy_memory_capacity": args.average_policy_memory_capacity,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_layers": _parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": _parse_int_tuple(args.regret_network_layers),
        "value_network_layers": _parse_int_tuple(args.value_network_layers),
        "reservoir_decode_chunk_size": args.reservoir_decode_chunk_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.smoke:
        smoke_defaults = {
            "num_iterations": 2,
            "num_traversals": 2,
            "num_val_fn_traversals": 2,
            "check_exploitability_every": 1,
            "memory_capacity": 128,
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
        }
        explicitly_set = {
            key for key, value in overrides.items() if value is not None
        }
        for key, value in smoke_defaults.items():
            if key not in explicitly_set:
                config[key] = value
    config["expected_final_nodes_touched"] = None
    config["smoke"] = bool(args.smoke)
    validate_config(config, smoke=args.smoke)
    return config


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def _fit_seed(training_seed: int, config: Mapping[str, object]) -> int:
    return int(config["fit_seed_offset"]) + int(training_seed)


def _run_seed(seed: int, config: dict, run_dir: Path) -> tuple[dict, list[dict]]:
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed_tf(seed)
    game = pyspiel.load_game(str(config["game_name"]))
    solver = make_escher_solver(game, config, run_seed=seed)

    _LOGGER.info(
        "Training source seed %s for %.2f active hours (iteration safety cap %s)",
        seed,
        float(config["training_wall_clock_seconds"]) / 3_600.0,
        config["num_iterations"],
    )
    training_started = time.perf_counter()
    _, _, convs, nodes, values, diagnostics = solver.solve(
        max_wall_clock_seconds=float(config["training_wall_clock_seconds"])
    )
    training_seconds = time.perf_counter() - training_started
    solve_summary = solver.get_last_solve_summary()
    if solve_summary is None:
        raise RuntimeError("ESCHER solver did not publish termination metadata")
    if not bool(config.get("smoke", False)) and not solve_summary[
        "hit_wall_clock_limit"
    ]:
        raise RuntimeError(
            f"Experiment {EXPERIMENT_ID} reached its iteration safety cap "
            "before the configured training-time endpoint"
        )
    source_metrics = exact_neural_policy_metrics(game, solver._policy_network)  # pylint: disable=protected-access
    source_final_iteration = int(solver._iteration)  # pylint: disable=protected-access
    serialized = list(solver.get_average_policy_memories())
    if len(serialized) > int(config["average_policy_memory_capacity"]):
        raise RuntimeError("Policy reservoir exceeded its configured capacity")

    _LOGGER.info("Decoding and freezing %s policy rows for seed %s", len(serialized), seed)
    frozen = decode_serialized_reservoir(
        serialized,
        solver._average_policy_feature_description,  # pylint: disable=protected-access
        chunk_size=int(config["reservoir_decode_chunk_size"]),
    )
    reservoir_path = seed_dir / str(config["frozen_reservoir_filename"])
    reservoir_manifest = save_frozen_reservoir(reservoir_path, frozen)
    # Reload the on-disk artifact so every arm demonstrably uses the saved,
    # identical source rather than an in-memory side channel.
    frozen = load_frozen_reservoir(reservoir_path)
    config_for_fit = dict(config, source_final_iteration=source_final_iteration)
    grouped = group_reservoir(frozen, iteration=source_final_iteration)
    empirical_metrics = exact_empirical_policy_metrics(game, grouped)

    source_row = {
        "seed": int(seed),
        "source_training_seconds": float(training_seconds),
        "source_active_training_seconds": float(
            solve_summary["active_training_seconds"]
        ),
        "source_training_budget_seconds": float(
            config["training_wall_clock_seconds"]
        ),
        "source_training_budget_overshoot_seconds": float(
            solve_summary["budget_overshoot_seconds"]
        ),
        "source_termination_reason": solve_summary["termination_reason"],
        "source_hit_wall_clock_limit": bool(
            solve_summary["hit_wall_clock_limit"]
        ),
        "source_completed_solve_passes": int(
            solve_summary["completed_solve_passes"]
        ),
        "source_final_iteration": source_final_iteration,
        "source_final_nodes_touched": int(solve_summary["nodes_touched"]),
        "source_last_checkpoint_nodes_touched": float(nodes[-1]),
        "source_final_recorded_nash_conv": float(convs[-1]),
        "source_final_recorded_exploitability": float(convs[-1]) / 2.0,
        "source_final_recorded_policy_value": float(values[-1]),
        "source_neural_exploitability_recomputed": source_metrics["exploitability"],
        "source_neural_policy_value_recomputed": source_metrics["policy_value"],
        "reservoir_rows": frozen.size,
        "unique_information_sets": grouped.size,
        "reservoir_capacity": int(config["average_policy_memory_capacity"]),
        "reservoir_fill_fraction": frozen.size / float(config["average_policy_memory_capacity"]),
        "empirical_reservoir_exploitability": empirical_metrics["exploitability"],
        "empirical_reservoir_policy_value": empirical_metrics["policy_value"],
        "empirical_missing_information_sets": empirical_metrics["missing_information_sets"],
        "source_neural_minus_empirical_gap": (
            source_metrics["exploitability"] - empirical_metrics["exploitability"]
        ),
        "trajectory_points": len(convs),
        "final_policy_loss": float(np.asarray(diagnostics["policy_loss"])[-1]),
        **{f"reservoir_{key}": value for key, value in reservoir_manifest.items() if key != "path"},
        "reservoir_path": str(reservoir_path.relative_to(run_dir)),
    }
    _write_json(seed_dir / "source_summary.json", source_row)

    # The frozen reservoir now owns the only source data needed by the audit;
    # release the regret/value solver state before fitting four independent nets.
    del serialized, solver
    cleanup_tensorflow_memory()
    gc.collect()

    fit_seed = _fit_seed(seed, config)
    base_weights = initial_policy_weights(
        config_for_fit, frozen, fit_seed=fit_seed
    )
    fit_rows = []
    for arm_id in ARM_ORDER:
        treatment = ARMS[arm_id]
        _LOGGER.info("Fitting seed %s arm %s", seed, arm_id)
        model, fit_diagnostics = fit_policy(
            frozen,
            grouped,
            config_for_fit,
            loss_name=str(treatment["loss"]),
            use_grouped_data=bool(treatment["grouped"]),
            example_multiplier=int(treatment["example_multiplier"]),
            fit_seed=fit_seed,
            base_weights=base_weights,
        )
        metrics = exact_neural_policy_metrics(game, model)
        weights_path = seed_dir / f"{arm_id}.weights.h5"
        model.save_weights(str(weights_path))
        fit_rows.append({
            "seed": int(seed),
            "fit_seed": int(fit_seed),
            "arm_id": arm_id,
            "arm_label": treatment["label"],
            "loss": treatment["loss"],
            "grouped": bool(treatment["grouped"]),
            "example_multiplier": int(treatment["example_multiplier"]),
            "exploitability": metrics["exploitability"],
            "nash_conv": metrics["nash_conv"],
            "policy_value": metrics["policy_value"],
            "policy_value_error": metrics["policy_value_error"],
            "empirical_reservoir_exploitability": empirical_metrics["exploitability"],
            "distillation_gap": metrics["exploitability"] - empirical_metrics["exploitability"],
            "reservoir_rows": frozen.size,
            "unique_information_sets": grouped.size,
            "weights_path": str(weights_path.relative_to(run_dir)),
            **fit_diagnostics,
        })
        del model
        cleanup_tensorflow_memory()

    _write_csv(seed_dir / "fit_metrics.csv", fit_rows)
    _write_json(seed_dir / "SUCCESS.json", {
        "status": "complete", "seed": int(seed), "num_arms": len(fit_rows)
    })
    del frozen, grouped, base_weights
    cleanup_tensorflow_memory()
    return source_row, fit_rows


def _summarise(source_rows: Sequence[Mapping], fit_rows: Sequence[Mapping]):
    summary_rows = []
    for arm_id in ARM_ORDER:
        rows = [row for row in fit_rows if row["arm_id"] == arm_id]
        for metric in (
            "exploitability",
            "distillation_gap",
            "policy_value_error",
            "fit_seconds",
        ):
            stats = safe_stats([float(row[metric]) for row in rows])
            summary_rows.append({
                "policy_id": arm_id,
                "policy_label": ARMS[arm_id]["label"],
                "metric": metric,
                **stats,
            })
    for policy_id, policy_label, metric_key in (
        ("empirical_reservoir_policy", "Empirical reservoir policy", "empirical_reservoir_exploitability"),
        ("source_neural_policy", "In-training neural policy", "source_neural_exploitability_recomputed"),
    ):
        stats = safe_stats([float(row[metric_key]) for row in source_rows])
        summary_rows.append({
            "policy_id": policy_id,
            "policy_label": policy_label,
            "metric": "exploitability",
            **stats,
        })
    return summary_rows


def _plot_metric(
    fit_rows: Sequence[Mapping],
    source_rows: Sequence[Mapping],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    arms = list(ARM_ORDER)
    means, errors = [], []
    for arm in arms:
        stats = safe_stats([
            float(row[metric]) for row in fit_rows if row["arm_id"] == arm
        ])
        means.append(stats["mean"])
        errors.append(stats["se"])
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(arms))
    ax.bar(x, means, yerr=errors, capsize=4, color="#2878B5")
    for index, arm in enumerate(arms):
        values = [float(row[metric]) for row in fit_rows if row["arm_id"] == arm]
        ax.scatter(np.full(len(values), index), values, color="black", s=22, zorder=3)
    if metric == "exploitability":
        empirical = safe_stats([
            float(row["empirical_reservoir_exploitability"])
            for row in source_rows
        ])
        ax.axhline(empirical["mean"], color="#E45756", linestyle="--", linewidth=2,
                   label="Mean empirical reservoir policy")
        ax.legend()
    ax.set_xticks(x, [ARMS[arm]["label"] for arm in arms], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _aggregate(run_dir: Path, source_rows: Sequence[Mapping], fit_rows: Sequence[Mapping]):
    summary_rows = _summarise(source_rows, fit_rows)
    _write_csv(run_dir / "source_seed_metrics.csv", source_rows)
    _write_csv(run_dir / "fit_metrics.csv", fit_rows)
    _write_csv(run_dir / "arm_summary.csv", summary_rows)
    _plot_metric(
        fit_rows,
        source_rows,
        metric="exploitability",
        ylabel="Exploitability (NashConv / 2)",
        title="Standard ESCHER frozen-reservoir distillation audit",
        output=run_dir / "exploitability_by_distillation_arm.png",
    )
    _plot_metric(
        fit_rows,
        source_rows,
        metric="distillation_gap",
        ylabel="Neural minus empirical-reservoir exploitability",
        title="Average-policy distillation gap",
        output=run_dir / "distillation_gap_by_arm.png",
    )
    result = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "num_completed_seeds": len(source_rows),
        "num_fit_rows": len(fit_rows),
        "arm_summary": summary_rows,
    }
    _write_json(run_dir / "aggregate_summary.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/frozen_policy_distillation_audit")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--average-policy-memory-capacity", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--regret-network-train-steps", type=int, default=None)
    parser.add_argument("--value-network-train-steps", type=int, default=None)
    parser.add_argument("--batch-size-regret", type=int, default=None)
    parser.add_argument("--batch-size-value", type=int, default=None)
    parser.add_argument("--batch-size-average-policy", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--regret-network-layers", default=None)
    parser.add_argument("--value-network-layers", default=None)
    parser.add_argument("--reservoir-decode-chunk-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = build_config(args)
    seeds = _parse_seeds(args.seeds, smoke=args.smoke)
    run_dir = create_run_dir(args.output_root, str(config["experiment_name"]))
    _configure_logging(run_dir, args.verbose)
    _write_json(run_dir / "experiment_metadata.json", {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "smoke": bool(args.smoke),
        "seeds": seeds,
        "config": config,
        "arms": ARMS,
    })
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Seeds: %s", seeds)

    source_rows, fit_rows, failures = [], [], []
    for index, seed in enumerate(seeds, start=1):
        _LOGGER.info("Starting seed %s (%s/%s)", seed, index, len(seeds))
        try:
            source_row, seed_fit_rows = _run_seed(seed, config, run_dir)
            source_rows.append(source_row)
            fit_rows.extend(seed_fit_rows)
            _aggregate(run_dir, source_rows, fit_rows)
        except Exception as exc:  # pragma: no cover - operational failure path
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failures.append({
                "seed": int(seed),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            _write_json(run_dir / "failed_seeds.json", failures)
            if not args.continue_on_error:
                return 1
        finally:
            cleanup_tensorflow_memory()

    if not source_rows:
        return 1
    result = _aggregate(run_dir, source_rows, fit_rows)
    if failures:
        result["status"] = "partial"
        result["failures"] = failures
        _write_json(run_dir / "aggregate_summary.json", result)
        return 2
    _LOGGER.info("Experiment %s complete: %s", EXPERIMENT_ID, run_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
