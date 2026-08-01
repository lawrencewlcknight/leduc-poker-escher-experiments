"""CLI for Experiment 43: long-horizon ESCHER checkpoint head-to-head."""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/escher_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from escher_poker.experiment_utils import create_run_dir, json_safe  # noqa: E402
from experiments.leduc_poker.escher_variant_config_utils import (  # noqa: E402
    make_variant_config,
)

from .analyse import run_analysis  # noqa: E402
from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, validate_config  # noqa: E402
from .train import run_training  # noqa: E402


_LOGGER = logging.getLogger(
    "escher_poker.experiment.final_candidate_checkpoint_head_to_head"
)


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format)
    handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def build_config(args, *, base_config=None) -> dict:
    config = deepcopy(DEFAULT_CONFIG if base_config is None else base_config)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "checkpoint_schedule": _parse_int_tuple(args.checkpoint_schedule),
        "check_exploitability_every": args.evaluation_interval,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "learning_rate": args.learning_rate,
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
        "compute_exploitability": args.compute_exploitability,
        "equivalence_epsilon": args.equivalence_epsilon,
        "annotate_heatmap": args.annotate_heatmap,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["target_solve_pass_count"] = int(config["num_iterations"]) + 1
    if args.iterations is not None:
        # The 15M-node estimate belongs only to the declared 1,300-iteration
        # design, not to CLI smoke-test horizons.
        config["expected_final_nodes_touched"] = None
    config = make_variant_config(config, {})
    validate_config(config)
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Experiment 28 ESCHER configuration to roughly 15M nodes, "
            "save five policies without restarting, and compare them exactly."
        )
    )
    parser.add_argument(
        "phase", nargs="?", default="all", choices=("all", "train", "analyse")
    )
    parser.add_argument("--output-root", default="outputs/final_candidate_checkpoint_head_to_head")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint-schedule", default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--equivalence-epsilon", type=float, default=None)
    parser.add_argument("--annotate-heatmap", type=_str2bool, default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    stored_metadata = None
    stored_config = None
    if args.phase == "analyse" and args.run_dir:
        metadata_path = Path(args.run_dir) / "experiment_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as handle:
                stored_metadata = json.load(handle)
            stored_config = stored_metadata.get("config")

    config = build_config(args, base_config=stored_config)
    seeds = (
        [int(seed) for seed in stored_metadata["seeds"]]
        if args.seeds is None and stored_metadata and stored_metadata.get("seeds")
        else _parse_seeds(args.seeds)
    )
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(args.output_root, str(config["experiment_name"]))
    _configure_logging(run_dir, args.verbose)

    metadata = dict(stored_metadata or {})
    metadata.update({
        "config": config,
        "seeds": seeds,
        "phase": args.phase,
        "training_protocol": "one uninterrupted solver trajectory per seed",
        "head_to_head_evaluation": "exact OpenSpiel expected value averaged over both seats",
        "statistical_unit": "independent training seed",
    })
    _write_json(run_dir / "experiment_metadata.json", metadata)
    _LOGGER.info("Phase: %s", args.phase)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Seeds: %s", seeds)
    _LOGGER.info("Configuration: %s", config)

    if args.phase in {"all", "train"}:
        outcome = run_training(
            config=config,
            seeds=seeds,
            run_dir=run_dir,
            continue_on_error=bool(args.continue_on_error),
        )
        required_count = len(config["checkpoint_schedule"])
        completed_seeds = [
            seed
            for seed in seeds
            if sum(
                1
                for row in outcome["metrics_rows"]
                if int(row["seed"]) == int(seed)
            )
            == required_count
        ]
        metadata.update({
            "completed_seeds": completed_seeds,
            "training_stage_metrics_csv": str(outcome["metrics_csv"]),
        })
        _write_json(run_dir / "experiment_metadata.json", metadata)
        _write_json(run_dir / "failed_seeds.json", outcome["failed"])
        if not completed_seeds:
            _LOGGER.error("No seed completed the full checkpoint schedule")
            return 1

    if args.phase in {"all", "analyse"}:
        snapshots_dir = run_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.glob("*.pkl")):
            _LOGGER.error("No snapshots found in %s", snapshots_dir)
            return 2
        outputs = run_analysis(
            config=config,
            run_dir=run_dir,
            snapshots_dir=snapshots_dir,
        )
        metadata["analysis_outputs"] = {key: str(value) for key, value in outputs.items()}
        _write_json(run_dir / "experiment_metadata.json", metadata)
        _LOGGER.info("Analysis outputs: %s", outputs)

    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
