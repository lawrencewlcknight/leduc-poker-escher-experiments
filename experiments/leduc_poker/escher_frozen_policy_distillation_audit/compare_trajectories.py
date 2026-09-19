"""Combine temporal source-policy trajectories from Experiments 45--49."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/escher_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402

from .trajectory import read_trajectory_rows  # noqa: E402


def _parse_experiment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=ANALYSIS_DIR_OR_CSV")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both label and path are required")
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "source_trajectory.csv"
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Trajectory file does not exist: {path}")
    return label.strip(), path


def _step_resample(rows: Sequence[Mapping], x_field: str, grid: np.ndarray):
    ordered = sorted(rows, key=lambda row: float(row[x_field]))
    x = np.asarray([float(row[x_field]) for row in ordered], dtype=np.float64)
    y = np.asarray(
        [float(row["exploitability"]) for row in ordered], dtype=np.float64
    )
    sampled = np.full(grid.shape, np.nan, dtype=np.float64)
    if not x.size:
        return sampled
    indices = np.searchsorted(x, grid, side="right") - 1
    valid = (indices >= 0) & (grid <= x[-1])
    sampled[valid] = y[indices[valid]]
    return sampled


def _make_grid(
    datasets: Sequence[tuple[str, Sequence[Mapping]]],
    *,
    x_field: str,
    time_step_hours: float,
    node_grid_points: int,
) -> np.ndarray:
    maximum = max(
        float(row[x_field])
        for _, rows in datasets
        for row in rows
    )
    if x_field == "training_hours":
        if time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive")
        regular = np.append(
            np.arange(0.0, maximum, time_step_hours), maximum
        )
    else:
        if node_grid_points < 2:
            raise ValueError("node_grid_points must be at least two")
        regular = np.linspace(0.0, maximum, node_grid_points)
    # Retain a regular comparison grid while guaranteeing that sparse or
    # overshooting experiments contribute at their actual evaluation times.
    observed = np.asarray([
        float(row[x_field])
        for _, rows in datasets
        for row in rows
    ])
    return np.unique(np.concatenate([regular, observed]))


def _summarise_on_grid(
    label: str,
    rows: Sequence[Mapping],
    *,
    x_field: str,
    grid: np.ndarray,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    seeds = sorted({int(row["seed"]) for row in rows})
    seed_curves = {
        seed: _step_resample(
            [row for row in rows if int(row["seed"]) == seed],
            x_field,
            grid,
        )
        for seed in seeds
    }
    matrix = np.vstack([seed_curves[seed] for seed in seeds])
    summary = []
    for index, x_value in enumerate(grid):
        values = matrix[:, index]
        finite = values[np.isfinite(values)]
        n = int(finite.size)
        std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
        summary.append({
            "comparison_label": label,
            "axis": x_field,
            "grid_index": index,
            "x_value": float(x_value),
            "mean_exploitability": float(np.mean(finite)) if n else np.nan,
            "std_exploitability": std if n else np.nan,
            "se_exploitability": float(std / np.sqrt(n)) if n else np.nan,
            "n_seeds": n,
            "configured_seed_count": len(seeds),
        })
    return summary, seed_curves


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(
    datasets: Sequence[tuple[str, Sequence[Mapping]]],
    *,
    x_field: str,
    xlabel: str,
    grid: np.ndarray,
    summaries: Mapping[str, Sequence[Mapping]],
    seed_curves: Mapping[str, Mapping[int, np.ndarray]],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    colours = plt.get_cmap("tab10")
    for experiment_index, (label, _) in enumerate(datasets):
        colour = colours(experiment_index % 10)
        for curve in seed_curves[label].values():
            ax.plot(
                grid, curve, color=colour, alpha=0.13, linewidth=0.8,
                drawstyle="steps-post"
            )
        summary = summaries[label]
        mean = np.asarray(
            [float(row["mean_exploitability"]) for row in summary]
        )
        se = np.asarray([float(row["se_exploitability"]) for row in summary])
        complete = np.asarray([
            int(row["n_seeds"]) == int(row["configured_seed_count"])
            for row in summary
        ])
        mean = np.where(complete, mean, np.nan)
        se = np.where(complete, se, np.nan)
        ax.plot(
            grid, mean, color=colour, linewidth=2.2,
            drawstyle="steps-post", label=label
        )
        ax.fill_between(
            grid, mean - se, mean + se,
            color=colour, alpha=0.16, step="post"
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, "ESCHER source-policy temporal comparison")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compare_trajectories(
    experiments: Sequence[tuple[str, Path]],
    output_dir: Path,
    *,
    time_step_hours: float = 0.25,
    node_grid_points: int = 201,
) -> dict:
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required")
    labels = [label for label, _ in experiments]
    if len(set(labels)) != len(labels):
        raise ValueError("Experiment labels must be unique")

    datasets = [(label, read_trajectory_rows(path)) for label, path in experiments]
    for label, rows in datasets:
        if not rows:
            raise ValueError(f"{label} has no trajectory rows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = [
        {"comparison_label": label, **dict(row)}
        for label, rows in datasets
        for row in rows
    ]
    _write_csv(output_dir / "combined_source_trajectory.csv", combined_rows)

    all_summary_rows = []
    outputs = {}
    for x_field, xlabel, filename in (
        (
            "training_hours",
            "Training time (hours)",
            "combined_exploitability_by_training_time.png",
        ),
        (
            "nodes_touched",
            "Nodes touched",
            "combined_exploitability_by_nodes.png",
        ),
    ):
        grid = _make_grid(
            datasets,
            x_field=x_field,
            time_step_hours=time_step_hours,
            node_grid_points=node_grid_points,
        )
        summaries, curves = {}, {}
        for label, rows in datasets:
            summaries[label], curves[label] = _summarise_on_grid(
                label, rows, x_field=x_field, grid=grid
            )
            all_summary_rows.extend(summaries[label])
        output = output_dir / filename
        _plot_comparison(
            datasets,
            x_field=x_field,
            xlabel=xlabel,
            grid=grid,
            summaries=summaries,
            seed_curves=curves,
            output=output,
        )
        outputs[x_field] = str(output)

    _write_csv(output_dir / "combined_trajectory_summary.csv", all_summary_rows)
    return {
        "experiments": labels,
        "num_raw_rows": len(combined_rows),
        "num_summary_rows": len(all_summary_rows),
        "outputs": outputs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        type=_parse_experiment,
        required=True,
        help="Repeat as LABEL=ANALYSIS_DIR_OR_SOURCE_TRAJECTORY_CSV",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-step-hours", type=float, default=0.25)
    parser.add_argument("--node-grid-points", type=int, default=201)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = compare_trajectories(
        args.experiment,
        args.output_dir,
        time_step_hours=args.time_step_hours,
        node_grid_points=args.node_grid_points,
    )
    print(result)


if __name__ == "__main__":  # pragma: no cover
    main()
