"""Cloud worker and aggregation entry points for Experiment 45."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence

from .config import ARM_ORDER, DEFAULT_SEEDS, EXPERIMENT_ID, EXPERIMENT_NAME
from .distillation import sha256
from .run import _aggregate, _write_json, main as run_experiment


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMOKE_SEEDS = (1234,)


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique")
    return seeds


def _read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_name(task_index: int, seeds: Sequence[int]) -> str:
    if task_index < 0 or task_index >= len(seeds):
        raise ValueError(
            f"Task index {task_index} is outside [0, {len(seeds) - 1}]"
        )
    return f"task_{task_index:03d}_seed_{int(seeds[task_index])}"


def _worker_result_is_valid(
    result_path: Path, *, seed: int, smoke: bool, repository_commit: str
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = _read_json(result_path)
        run_dir = result_path.parent / result["run_dir"]
        source = _read_json(run_dir / "seed_{}".format(seed) / "source_summary.json")
        fit_rows = _read_csv(run_dir / "seed_{}".format(seed) / "fit_metrics.csv")
        reservoir = run_dir / str(source["reservoir_path"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        result.get("status") == "complete"
        and int(result.get("experiment_id", -1)) == EXPERIMENT_ID
        and int(result.get("seed", -1)) == int(seed)
        and bool(result.get("smoke")) == bool(smoke)
        and result.get("repository_commit") == repository_commit
        and (run_dir / f"seed_{seed}" / "SUCCESS.json").is_file()
        and reservoir.is_file()
        and sha256(reservoir) == source.get("reservoir_sha256")
        and {row.get("arm_id") for row in fit_rows} == set(ARM_ORDER)
        and len(fit_rows) == len(ARM_ORDER)
    )


def run_worker(
    *, task_index: int, seeds: Sequence[int], output_root: Path,
    smoke: bool, resume: bool,
) -> dict:
    name = task_name(task_index, seeds)
    seed = int(seeds[task_index])
    task_dir = Path(output_root) / "workers" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "worker_result.json"
    repository_commit = _repository_commit()
    if resume and _worker_result_is_valid(
        result_path,
        seed=seed,
        smoke=smoke,
        repository_commit=repository_commit,
    ):
        return _read_json(result_path)

    arguments = ["--output-root", str(task_dir), "--seeds", str(seed)]
    if smoke:
        arguments.append("--smoke")
    exit_code = run_experiment(arguments)
    if exit_code != 0:
        raise RuntimeError(
            f"Experiment 45 worker {name} exited with status {exit_code}"
        )

    candidates = sorted(
        task_dir.glob(f"{EXPERIMENT_NAME}_*"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"Experiment 45 worker {name} produced no run directory")
    run_dir = candidates[0]
    seed_dir = run_dir / f"seed_{seed}"
    source_summary = seed_dir / "source_summary.json"
    fit_metrics = seed_dir / "fit_metrics.csv"
    if not (seed_dir / "SUCCESS.json").is_file():
        raise RuntimeError(f"Experiment 45 worker {name} has no success marker")
    fit_rows = _read_csv(fit_metrics)
    source = _read_json(source_summary)
    reservoir = run_dir / str(source["reservoir_path"])
    if (
        len(fit_rows) != len(ARM_ORDER)
        or {row.get("arm_id") for row in fit_rows} != set(ARM_ORDER)
        or not source_summary.is_file()
        or not reservoir.is_file()
        or sha256(reservoir) != source.get("reservoir_sha256")
    ):
        raise RuntimeError(f"Experiment 45 worker {name} has incomplete outputs")

    result = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "task_index": int(task_index),
        "task_name": name,
        "seed": seed,
        "smoke": bool(smoke),
        "repository_commit": repository_commit,
        "run_dir": str(run_dir.relative_to(task_dir)),
        "artifacts": {
            "source_summary": str(source_summary.relative_to(task_dir)),
            "fit_metrics": str(fit_metrics.relative_to(task_dir)),
            "frozen_reservoir": str(
                reservoir.relative_to(task_dir)
            ),
        },
    }
    _write_json(result_path, result)
    return result


def aggregate_workers(
    *, workers_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    expected = {int(seed) for seed in seeds}
    found: dict[int, tuple[Path, Mapping]] = {}
    for result_path in Path(workers_root).rglob("worker_result.json"):
        result = _read_json(result_path)
        seed = int(result.get("seed", -1))
        if seed in found:
            raise ValueError(f"Duplicate Experiment 45 worker for seed {seed}")
        if (
            result.get("status") != "complete"
            or int(result.get("experiment_id", -1)) != EXPERIMENT_ID
            or bool(result.get("smoke")) != bool(smoke)
        ):
            raise ValueError(f"Incomplete or mismatched worker: {result_path}")
        found[seed] = (result_path, result)
    if set(found) != expected:
        raise ValueError(
            f"Experiment 45 workers differ; missing={sorted(expected-set(found))}, "
            f"extra={sorted(set(found)-expected)}"
        )
    commits = {str(result["repository_commit"]) for _, result in found.values()}
    if len(commits) != 1:
        raise ValueError(f"Workers used different commits: {sorted(commits)}")

    source_rows, fit_rows = [], []
    for seed in seeds:
        result_path, result = found[int(seed)]
        task_dir = result_path.parent
        run_dir = task_dir / str(result["run_dir"])
        seed_dir = run_dir / f"seed_{int(seed)}"
        if not (seed_dir / "SUCCESS.json").is_file():
            raise ValueError(f"Missing success marker for seed {seed}")
        source_row = _read_json(seed_dir / "source_summary.json")
        rows = _read_csv(seed_dir / "fit_metrics.csv")
        reservoir = run_dir / str(source_row["reservoir_path"])
        if int(source_row.get("seed", -1)) != int(seed):
            raise ValueError(f"Source summary has the wrong seed for {seed}")
        if (
            len(rows) != len(ARM_ORDER)
            or {row.get("arm_id") for row in rows} != set(ARM_ORDER)
            or {int(row.get("seed", -1)) for row in rows} != {int(seed)}
            or not reservoir.is_file()
            or sha256(reservoir) != source_row.get("reservoir_sha256")
        ):
            raise ValueError(f"Fit metrics are incomplete for seed {seed}")
        source_rows.append(source_row)
        fit_rows.extend(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _aggregate(output_dir, source_rows, fit_rows)
    result.update({
        "smoke": bool(smoke),
        "production_seeds": [int(seed) for seed in seeds],
        "repository_commit": next(iter(commits)),
        "execution_contract": "one isolated seed task per VM",
    })
    _write_json(output_dir / "aggregate_summary.json", result)
    return result


def _normalise_seeds(args) -> tuple[int, ...]:
    if args.smoke and args.seeds == tuple(DEFAULT_SEEDS):
        return SMOKE_SEEDS
    return tuple(args.seeds)


def _cmd_worker(args) -> None:
    seeds = _normalise_seeds(args)
    result = run_worker(
        task_index=args.task_index,
        seeds=seeds,
        output_root=args.output_root,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


def _cmd_aggregate(args) -> None:
    seeds = _normalise_seeds(args)
    result = aggregate_workers(
        workers_root=Path(args.output_root) / "workers",
        seeds=seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args) -> None:
    output_root = Path(args.output_root).resolve()
    run_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        output_root=output_root,
        smoke=True,
        resume=args.resume,
    )
    result = aggregate_workers(
        workers_root=output_root / "workers",
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args) -> None:
    seeds = _normalise_seeds(args)
    print(json.dumps({
        "tasks": [
            {"task_index": index, "task_name": task_name(index, seeds), "seed": seed}
            for index, seed in enumerate(seeds)
        ],
        "parallelism": len(seeds),
    }, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("worker", "aggregate", "schedule"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-root", type=Path, required=command != "schedule")
        child.add_argument("--seeds", type=_parse_seeds, default=tuple(DEFAULT_SEEDS))
        child.add_argument("--smoke", action="store_true")
        if command == "worker":
            child.add_argument("--task-index", type=int, required=True)
            child.add_argument("--no-resume", dest="resume", action="store_false")
            child.set_defaults(func=_cmd_worker, resume=True)
        elif command == "aggregate":
            child.set_defaults(func=_cmd_aggregate)
        else:
            child.set_defaults(func=_cmd_schedule)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
