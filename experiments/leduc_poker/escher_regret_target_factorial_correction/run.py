"""Run Experiment 37's staged 2x2 ESCHER correction factorial."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XDG_CACHE_HOME", str((Path("outputs") / ".cache").resolve()))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path("outputs") / ".matplotlib_cache").resolve()),
)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.experiment_utils import (  # noqa: E402
    cleanup_tensorflow_memory,
    create_run_dir,
    json_safe,
    run_single_seed_variant,
)
from escher_poker.factorial_metrics import (  # noqa: E402
    factorial_effect_rows,
    metric_at_target_nodes,
    rank_screening_variants,
)
from experiments.leduc_poker.escher_variant_ablation_runner import (  # noqa: E402
    DEFAULT_PAIRED_DELTA_FIELDS,
    DEFAULT_SUMMARY_HP_FIELDS,
    _aggregate_by_variant,
    _augment_summary,
    _numeric_summary,
    _paired_rows,
    _paired_summary,
    _plot_curves,
    _plot_final_exploitability,
    _safe_stem,
    _write_csv,
    _write_json,
)
from experiments.leduc_poker.escher_variant_config_utils import (  # noqa: E402
    make_variant_config,
    parse_variant_ids,
    variant_lookup,
)

from .config import (  # noqa: E402
    BASELINE_VARIANT_ID,
    BOTH_VARIANT_ID,
    CONFIRMATION_SEEDS,
    CONFIRMATION_TOP_K,
    DEFAULT_CONFIG,
    MEANINGFUL_SUCCESS_THRESHOLD,
    POLICY_ONLY_VARIANT_ID,
    SCALE_ONLY_VARIANT_ID,
    SCREENING_SEEDS,
    TARGET_NODES,
    VARIANTS,
)


WORKER_MODULE = (
    "experiments.leduc_poker.escher_regret_target_factorial_correction.run"
)
NODE_MATCHED_METRIC = "exploitability_at_target_nodes"
PAIRED_DELTA_FIELDS = list(DEFAULT_PAIRED_DELTA_FIELDS) + [NODE_MATCHED_METRIC]


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_seeds(value: Optional[str], default: Sequence[int]) -> List[int]:
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_layers(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(json_safe(payload), sort_keys=True))
        output.write("\n")


def _apply_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    overrides = {
        "num_iterations": args.iterations,
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
        "policy_network_layers": _parse_layers(args.policy_network_layers),
        "regret_network_layers": _parse_layers(args.regret_network_layers),
        "value_network_layers": _parse_layers(args.value_network_layers),
        "compute_exploitability": args.compute_exploitability,
        "save_final_checkpoints": args.save_final_checkpoints,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _augment_node_matched_summary(result, target_nodes, success_threshold, stage):
    node_metric = metric_at_target_nodes(result["curves"], target_nodes)
    value = float(node_metric["value"])
    result["summary"].update({
        "stage": stage,
        "target_nodes": float(target_nodes),
        NODE_MATCHED_METRIC: value,
        "node_matched_evaluation_nodes": float(node_metric["evaluation_nodes"]),
        "target_nodes_reached": bool(node_metric["target_nodes_reached"]),
        "absolute_target_node_gap": float(node_metric["node_gap"]),
        "meaningful_success_threshold": float(success_threshold),
        "meaningful_success": bool(
            np.isfinite(value) and value < float(success_threshold)
        ),
        "distance_to_meaningful_success": (
            value - float(success_threshold) if np.isfinite(value) else np.nan
        ),
    })
    for row in result["curves"]:
        row["stage"] = stage
    return result


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    result = run_single_seed_variant(
        int(payload["seed"]),
        payload["config"],
        export_dir=payload.get("export_dir"),
    )
    _write_json(output_path, result)
    return 0


def _run_isolated(seed, config, stage_dir, stage):
    stem = f"{_safe_stem(stage)}__{_safe_stem(config['variant_id'])}_seed_{seed}"
    worker_input = stage_dir / "worker_inputs" / f"{stem}.json"
    worker_output = stage_dir / "worker_results" / f"{stem}.json"
    worker_log = stage_dir / "worker_logs" / f"{stem}.log"
    _write_json(worker_input, {
        "seed": int(seed),
        "config": config,
        "export_dir": str(stage_dir),
    })
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--worker-input-json",
        str(worker_input),
        "--worker-output-json",
        str(worker_output),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(worker_log, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed with exit code {completed.returncode}; see {worker_log}."
        )
    if not worker_output.exists():
        raise RuntimeError(f"Worker completed without writing {worker_output}.")
    with open(worker_output, "r", encoding="utf-8") as source:
        return json.load(source)


def _run_stage(
    *,
    stage,
    seeds,
    variants,
    base_config,
    run_dir,
    target_nodes,
    success_threshold,
    subprocess_isolation,
    continue_on_error,
):
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    total = len(seeds) * len(variants)
    with tqdm(total=total, desc=f"Factorial {stage}") as progress:
        for seed in seeds:
            for variant in variants:
                config = make_variant_config(base_config, variant)
                config["stage"] = stage
                try:
                    if subprocess_isolation:
                        result = _run_isolated(seed, config, stage_dir, stage)
                    else:
                        result = run_single_seed_variant(
                            seed,
                            config,
                            export_dir=stage_dir,
                        )
                    result = _augment_node_matched_summary(
                        result,
                        target_nodes,
                        success_threshold,
                        stage,
                    )
                    result["summary"] = _augment_summary(
                        result["summary"],
                        config,
                        DEFAULT_SUMMARY_HP_FIELDS,
                        {
                            "factor_policy_weighted_q": (
                                "policy_weighted_q_correction"
                            ),
                            "factor_scale_only_normalization": (
                                "scale_only_normalization"
                            ),
                        },
                    )
                    results.append(result)
                    _append_jsonl(
                        stage_dir / "partial_variant_seed_summary.jsonl",
                        result["summary"],
                    )
                    for row in result["curves"]:
                        _append_jsonl(
                            stage_dir / "partial_checkpoint_curves.jsonl",
                            row,
                        )
                except Exception as exc:  # pragma: no cover - operational path
                    failures.append({
                        "stage": stage,
                        "seed": int(seed),
                        "variant_id": variant["variant_id"],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })
                    _write_json(stage_dir / "failed_runs.json", failures)
                    if not continue_on_error:
                        return results, failures
                finally:
                    cleanup_tensorflow_memory()
                    progress.update(1)
    return results, failures


def _plot_node_matched_metric(stage_dir, summary_rows, variants, threshold, stage):
    means = []
    standard_errors = []
    labels = []
    for variant in variants:
        variant_id = str(variant["variant_id"])
        values = np.asarray([
            row.get(NODE_MATCHED_METRIC, np.nan)
            for row in summary_rows
            if str(row["variant_id"]) == variant_id
        ], dtype=np.float64)
        values = values[np.isfinite(values)]
        means.append(float(np.mean(values)) if values.size else np.nan)
        standard_errors.append(
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        )
        labels.append(str(variant["variant_label"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=standard_errors, capsize=4)
    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Meaningful success threshold ({threshold:g})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Exploitability near one million nodes")
    set_chart_title(ax, f"ESCHER correction factorial: {stage} node-matched result")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        stage_dir / "exploitability_at_target_nodes_by_variant.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _export_stage(stage_dir, results, variants, baseline_variant_id, plot_title):
    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    paired_rows = _paired_rows(
        summary_rows,
        baseline_variant_id,
        PAIRED_DELTA_FIELDS,
    )
    _write_csv(stage_dir / "variant_seed_summary.csv", summary_rows)
    _write_csv(stage_dir / "checkpoint_curves.csv", curve_rows)
    _write_csv(stage_dir / "paired_differences_vs_baseline.csv", paired_rows)
    _write_json(
        stage_dir / "aggregate_summary.json",
        _aggregate_by_variant(summary_rows, variants),
    )
    _write_json(
        stage_dir / "paired_difference_summary.json",
        _paired_summary(paired_rows, PAIRED_DELTA_FIELDS),
    )
    _plot_final_exploitability(stage_dir, summary_rows, variants, plot_title)
    _plot_curves(stage_dir, curve_rows, variants, plot_title)
    return summary_rows, curve_rows, paired_rows


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run the staged Experiment 37 ESCHER correction factorial."
    )
    parser.add_argument(
        "--output-root",
        default="outputs/regret_target_factorial_correction",
    )
    parser.add_argument("--screening-seeds", default=None)
    parser.add_argument("--confirmation-seeds", default=None)
    parser.add_argument("--variant-ids", default=None)
    parser.add_argument("--confirmation-top-k", type=int, default=CONFIRMATION_TOP_K)
    parser.add_argument("--target-nodes", type=float, default=TARGET_NODES)
    parser.add_argument(
        "--meaningful-success-threshold",
        type=float,
        default=MEANINGFUL_SUCCESS_THRESHOLD,
    )
    parser.add_argument("--screening-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
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
    parser.add_argument("--save-final-checkpoints", type=_str2bool, default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument("--disable-subprocess-isolation", action="store_true")
    parser.add_argument("--worker-input-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker JSON paths are required.")
        return _run_worker(
            Path(args.worker_input_json),
            Path(args.worker_output_json),
        )

    screening_seeds = _parse_seeds(args.screening_seeds, SCREENING_SEEDS)
    confirmation_seeds = _parse_seeds(args.confirmation_seeds, CONFIRMATION_SEEDS)
    if not args.screening_only and set(screening_seeds) & set(confirmation_seeds):
        raise ValueError("Screening and confirmation seeds must be disjoint.")
    if args.confirmation_top_k < 1:
        raise ValueError("confirmation_top_k must be at least one.")
    if args.target_nodes <= 0.0:
        raise ValueError("target_nodes must be positive.")

    selected_ids = parse_variant_ids(args.variant_ids, VARIANTS)
    lookup = variant_lookup(VARIANTS)
    unknown = [variant_id for variant_id in selected_ids if variant_id not in lookup]
    if unknown:
        raise ValueError(f"Unknown variant id(s): {unknown}")
    screening_variants = [lookup[variant_id] for variant_id in selected_ids]

    base_config = _apply_overrides(deepcopy(DEFAULT_CONFIG), args)
    base_config["target_nodes"] = float(args.target_nodes)
    base_config["meaningful_success_threshold"] = float(
        args.meaningful_success_threshold
    )
    base_config["confirmation_top_k"] = int(args.confirmation_top_k)
    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _write_json(run_dir / "experiment_metadata.json", {
        "base_config": base_config,
        "screening_seeds": screening_seeds,
        "confirmation_seeds": confirmation_seeds,
        "screening_variant_ids": selected_ids,
        "confirmation_top_k": int(args.confirmation_top_k),
        "target_nodes": float(args.target_nodes),
        "meaningful_success_threshold": float(args.meaningful_success_threshold),
        "subprocess_isolation": not args.disable_subprocess_isolation,
    })

    screening_results, screening_failures = _run_stage(
        stage="screening",
        seeds=screening_seeds,
        variants=screening_variants,
        base_config=base_config,
        run_dir=run_dir,
        target_nodes=args.target_nodes,
        success_threshold=args.meaningful_success_threshold,
        subprocess_isolation=not args.disable_subprocess_isolation,
        continue_on_error=args.continue_on_error,
    )
    if not screening_results:
        return 1
    screening_rows, _, _ = _export_stage(
        run_dir / "screening",
        screening_results,
        screening_variants,
        BASELINE_VARIANT_ID,
        "ESCHER correction factorial screening",
    )
    _plot_node_matched_metric(
        run_dir / "screening",
        screening_rows,
        screening_variants,
        args.meaningful_success_threshold,
        "screening",
    )
    ranking, confirmation_ids = rank_screening_variants(
        screening_rows,
        baseline_variant_id=BASELINE_VARIANT_ID,
        confirmation_top_k=args.confirmation_top_k,
    )
    _write_csv(run_dir / "screening" / "screening_ranking.csv", ranking)
    _write_json(run_dir / "screening" / "confirmation_selection.json", {
        "selected_variant_ids": confirmation_ids,
        "selection_metric": NODE_MATCHED_METRIC,
        "confirmation_top_k_treatments": int(args.confirmation_top_k),
        "baseline_always_included": BASELINE_VARIANT_ID,
    })
    factorial_rows = factorial_effect_rows(
        screening_rows,
        baseline_variant_id=BASELINE_VARIANT_ID,
        policy_only_variant_id=POLICY_ONLY_VARIANT_ID,
        scale_only_variant_id=SCALE_ONLY_VARIANT_ID,
        both_variant_id=BOTH_VARIANT_ID,
    )
    _write_csv(
        run_dir / "screening" / "factorial_effects_by_seed.csv",
        factorial_rows,
    )
    _write_json(
        run_dir / "screening" / "factorial_effect_summary.json",
        _numeric_summary(factorial_rows),
    )

    confirmation_results = []
    confirmation_failures = []
    confirmation_variants = [
        lookup[variant_id]
        for variant_id in confirmation_ids
        if variant_id in lookup
    ]
    if not args.screening_only:
        confirmation_results, confirmation_failures = _run_stage(
            stage="confirmation",
            seeds=confirmation_seeds,
            variants=confirmation_variants,
            base_config=base_config,
            run_dir=run_dir,
            target_nodes=args.target_nodes,
            success_threshold=args.meaningful_success_threshold,
            subprocess_isolation=not args.disable_subprocess_isolation,
            continue_on_error=args.continue_on_error,
        )
        if confirmation_results:
            confirmation_rows, _, _ = _export_stage(
                run_dir / "confirmation",
                confirmation_results,
                confirmation_variants,
                BASELINE_VARIANT_ID,
                "ESCHER correction factorial confirmation",
            )
            _plot_node_matched_metric(
                run_dir / "confirmation",
                confirmation_rows,
                confirmation_variants,
                args.meaningful_success_threshold,
                "confirmation",
            )

    _write_json(run_dir / "summary.json", {
        "screening_ranking": ranking,
        "confirmation_variant_ids": confirmation_ids,
        "screening_factorial_effect_summary": _numeric_summary(factorial_rows),
        "screening_failures": screening_failures,
        "confirmation_failures": confirmation_failures,
        "meaningful_success_definition": {
            "metric": NODE_MATCHED_METRIC,
            "target_nodes": float(args.target_nodes),
            "threshold": float(args.meaningful_success_threshold),
            "comparison": "strictly_less_than",
        },
    })
    print(f"Saved Experiment 37 outputs to: {run_dir.resolve()}")
    return 0 if not screening_failures and not confirmation_failures else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
