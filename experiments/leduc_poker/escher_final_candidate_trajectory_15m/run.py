"""Run Experiment 44 and persist the complete ESCHER evaluation trajectory."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/escher_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from escher_poker.experiment_utils import (  # noqa: E402
    cleanup_tensorflow_memory,
    create_run_dir,
    export_metadata,
    export_seed_summary,
    export_trajectory_history,
    run_single_seed,
)
from escher_poker.plotting import plot_diagnostics, plot_multiseed_results  # noqa: E402
from experiments.leduc_poker.escher_variant_config_utils import (  # noqa: E402
    make_variant_config,
)

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    expected_trajectory_points,
    validate_config,
)


_LOGGER = logging.getLogger("escher_poker.experiment.final_candidate_trajectory_15m")


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_seeds(value: Optional[str]) -> List[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "check_exploitability_every": args.evaluation_interval,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "memory_capacity": args.memory_capacity,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "policy_network_layers": _parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": _parse_int_tuple(args.regret_network_layers),
        "value_network_layers": _parse_int_tuple(args.value_network_layers),
        "save_final_checkpoints": args.save_final_checkpoints,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["target_solve_pass_count"] = int(config["num_iterations"]) + 1
    config["expected_trajectory_points"] = expected_trajectory_points(config)
    if (
        args.iterations is not None
        or args.traversals is not None
        or args.value_traversals is not None
    ):
        config["expected_final_nodes_touched"] = None
        config["trajectory_auc_end_nodes"] = None
    config = make_variant_config(config, {})
    validate_config(config)
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected ESCHER configuration to approximately 15M nodes "
            "and save every exploitability evaluation."
        )
    )
    parser.add_argument(
        "--output-root", default="outputs/final_candidate_trajectory_15m"
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-regret", type=int, default=None)
    parser.add_argument("--batch-size-value", type=int, default=None)
    parser.add_argument("--batch-size-average-policy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--regret-network-train-steps", type=int, default=None)
    parser.add_argument("--value-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--regret-network-layers", default=None)
    parser.add_argument("--value-network-layers", default=None)
    parser.add_argument("--save-final-checkpoints", type=_str2bool, default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)
    handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = build_config(args)
    seeds = _parse_seeds(args.seeds)
    run_dir = create_run_dir(args.output_root, str(config["experiment_name"]))
    _configure_logging(run_dir, args.verbose)
    export_metadata(run_dir, config, seeds)

    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Seeds: %s", seeds)
    _LOGGER.info(
        "Saving %s trajectory points per completed seed",
        config["expected_trajectory_points"],
    )

    results = []
    failures = []
    for seed_index, seed in enumerate(seeds, start=1):
        _LOGGER.info("Starting seed %s (%s/%s)", seed, seed_index, len(seeds))
        try:
            result = run_single_seed(seed, config, export_dir=run_dir)
            expected_points = int(config["expected_trajectory_points"])
            actual_points = len(result["iterations"])
            if actual_points != expected_points:
                raise RuntimeError(
                    f"Seed {seed} returned {actual_points} trajectory points; "
                    f"expected {expected_points}"
                )
            results.append(result)
            # Rewrite after each successful seed so completed trajectories
            # survive a later seed or machine failure.
            export_trajectory_history(run_dir, results)
            export_seed_summary(run_dir, results)
        except Exception as exc:  # pragma: no cover - operational failure path
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failures.append({
                "seed": int(seed),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            if not args.continue_on_error:
                break
        finally:
            cleanup_tensorflow_memory()

    if failures:
        with open(run_dir / "failed_seeds.json", "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2)
    if not results:
        _LOGGER.error("No seeds completed successfully")
        return 1

    aggregate = export_seed_summary(run_dir, results)
    export_trajectory_history(run_dir, results)
    plot_multiseed_results(
        results,
        run_dir,
        average_policy_value_target=float(config["average_policy_value_target"]),
    )
    plot_diagnostics(results, run_dir)
    _LOGGER.info("Final exploitability: %s", aggregate["final_exploitability"])
    _LOGGER.info(
        "14M-15M final-window mean exploitability: %s",
        aggregate["final_window_mean_exploitability"],
    )
    _LOGGER.info(
        "Normalised exploitability AUC over 0-15M nodes: %s",
        aggregate["normalised_auc_exploitability_0_to_target_nodes"],
    )
    _LOGGER.info("Saved all outputs to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
