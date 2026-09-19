"""Trajectory artifacts shared by Experiments 45 and their derivatives."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from escher_poker.chart_titles import set_chart_title


DIAGNOSTIC_FIELDS = (
    "policy_loss",
    "value_loss",
    "value_test_loss",
    "regret_loss_player_0",
    "regret_loss_player_1",
    "average_policy_buffer_size",
    "regret_buffer_size_player_0",
    "regret_buffer_size_player_1",
    "value_buffer_size",
    "value_test_buffer_size",
    "peak_rss_mb",
    "cumulative_experience_collection_seconds",
)

TRAJECTORY_FIELDS = (
    "experiment_id",
    "experiment_name",
    "seed",
    "checkpoint_index",
    "is_final_policy_fit",
    "iteration",
    "solver_iteration",
    "nodes_touched",
    "wall_clock_seconds",
    "training_hours",
    "nash_conv",
    "exploitability",
    "average_policy_value",
    *DIAGNOSTIC_FIELDS,
)


def _diagnostic_at(diagnostics: Mapping, name: str, index: int):
    values = diagnostics.get(name, ())
    if index >= len(values):
        return np.nan
    value = values[index]
    if "buffer_size" in name:
        return int(value)
    return float(value)


def build_trajectory_rows(
    *,
    experiment_id: int,
    experiment_name: str,
    seed: int,
    nash_convs: Sequence[float],
    nodes_touched: Sequence[float],
    average_policy_values: Sequence[float],
    diagnostics: Mapping,
) -> list[dict]:
    """Build one canonical row per in-training exploitability evaluation."""
    lengths = {
        len(nash_convs),
        len(nodes_touched),
        len(average_policy_values),
        len(diagnostics.get("wall_clock_seconds", ())),
    }
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(
            "Exploitability, node, value and wall-clock trajectories must have "
            "the same positive length"
        )

    iterations = diagnostics.get("iteration", ())
    solver_iterations = diagnostics.get("solver_iteration", ())
    point_count = len(nash_convs)
    if len(iterations) != point_count or len(solver_iterations) != point_count:
        raise ValueError("Iteration diagnostics do not align with evaluations")

    rows = []
    for index in range(point_count):
        wall_clock_seconds = float(diagnostics["wall_clock_seconds"][index])
        nash_conv = float(nash_convs[index])
        row = {
            "experiment_id": int(experiment_id),
            "experiment_name": str(experiment_name),
            "seed": int(seed),
            "checkpoint_index": int(index),
            "is_final_policy_fit": False,
            "iteration": int(iterations[index]),
            "solver_iteration": int(solver_iterations[index]),
            "nodes_touched": float(nodes_touched[index]),
            "wall_clock_seconds": wall_clock_seconds,
            "training_hours": wall_clock_seconds / 3_600.0,
            "nash_conv": nash_conv,
            "exploitability": nash_conv / 2.0,
            "average_policy_value": float(average_policy_values[index]),
        }
        row.update({
            name: _diagnostic_at(diagnostics, name, index)
            for name in DIAGNOSTIC_FIELDS
        })
        rows.append(row)
    return rows


def build_final_policy_row(
    *,
    experiment_id: int,
    experiment_name: str,
    seed: int,
    checkpoint_index: int,
    iteration: int,
    nodes_touched: float,
    wall_clock_seconds: float,
    metrics: Mapping[str, float],
    final_policy_loss: float,
    diagnostics: Mapping,
) -> dict:
    """Record the playable policy fitted once after the timed solve loop."""
    row = {
        "experiment_id": int(experiment_id),
        "experiment_name": str(experiment_name),
        "seed": int(seed),
        "checkpoint_index": int(checkpoint_index),
        "is_final_policy_fit": True,
        "iteration": int(iteration),
        "solver_iteration": int(iteration),
        "nodes_touched": float(nodes_touched),
        "wall_clock_seconds": float(wall_clock_seconds),
        "training_hours": float(wall_clock_seconds) / 3_600.0,
        "nash_conv": float(metrics["nash_conv"]),
        "exploitability": float(metrics["exploitability"]),
        "average_policy_value": float(metrics["policy_value"]),
    }
    for name in DIAGNOSTIC_FIELDS:
        if name == "policy_loss":
            row[name] = float(final_policy_loss)
            continue
        values = diagnostics.get(name, ())
        row[name] = (
            _diagnostic_at(diagnostics, name, len(values) - 1)
            if len(values)
            else np.nan
        )
    return row


def write_trajectory_rows(path: Path, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in TRAJECTORY_FIELDS})


def read_trajectory_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats(values: Sequence[float]) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    if not n:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "n": 0}
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(finite)),
        "std": std,
        "se": float(std / np.sqrt(n)),
        "n": n,
    }


def summarise_trajectory_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Summarise variable-length seed curves by shared checkpoint index."""
    if not rows:
        raise ValueError("At least one trajectory row is required")
    experiment_ids = {int(row["experiment_id"]) for row in rows}
    experiment_names = {str(row["experiment_name"]) for row in rows}
    if len(experiment_ids) != 1 or len(experiment_names) != 1:
        raise ValueError("One aggregate may contain only one experiment")

    summary = []
    intermediate_indices = sorted({
        int(row["checkpoint_index"])
        for row in rows
        if str(row.get("is_final_policy_fit", "False")).lower() != "true"
    })
    group_keys = [(False, index) for index in intermediate_indices]
    if any(
        str(row.get("is_final_policy_fit", "False")).lower() == "true"
        for row in rows
    ):
        group_keys.append((True, -1))
    for is_final_policy_fit, checkpoint_index in group_keys:
        checkpoint_rows = [
            row for row in rows
            if (
                str(row.get("is_final_policy_fit", "False")).lower() == "true"
            ) == is_final_policy_fit
            and (
                is_final_policy_fit
                or int(row["checkpoint_index"]) == checkpoint_index
            )
        ]
        exploitability = _stats([
            float(row["exploitability"]) for row in checkpoint_rows
        ])
        time_values = _stats([
            float(row["training_hours"]) for row in checkpoint_rows
        ])
        node_values = _stats([
            float(row["nodes_touched"]) for row in checkpoint_rows
        ])
        summary.append({
            "experiment_id": next(iter(experiment_ids)),
            "experiment_name": next(iter(experiment_names)),
            "checkpoint_index": checkpoint_index,
            "is_final_policy_fit": is_final_policy_fit,
            "mean_iteration": float(np.mean([
                float(row["iteration"]) for row in checkpoint_rows
            ])),
            "mean_solver_iteration": float(np.mean([
                float(row["solver_iteration"]) for row in checkpoint_rows
            ])),
            "mean_nodes_touched": node_values["mean"],
            "std_nodes_touched": node_values["std"],
            "se_nodes_touched": node_values["se"],
            "mean_training_hours": time_values["mean"],
            "std_training_hours": time_values["std"],
            "se_training_hours": time_values["se"],
            "mean_exploitability": exploitability["mean"],
            "std_exploitability": exploitability["std"],
            "se_exploitability": exploitability["se"],
            "n_seeds": exploitability["n"],
        })
    return summary


def write_trajectory_summary(path: Path, rows: Sequence[Mapping]) -> list[dict]:
    summary = summarise_trajectory_rows(rows)
    fields = tuple(summary[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot_trajectory(
    rows: Sequence[Mapping],
    summary: Sequence[Mapping],
    *,
    x_field: str,
    summary_x_field: str,
    xlabel: str,
    output: Path,
) -> None:
    """Plot raw seed paths and the checkpoint-aligned cross-seed mean."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: float(row[x_field]),
        )
        ax.plot(
            [float(row[x_field]) for row in seed_rows],
            [float(row["exploitability"]) for row in seed_rows],
            color="#2878B5",
            alpha=0.24,
            linewidth=1.0,
            drawstyle="steps-post",
        )
    x = np.asarray([float(row[summary_x_field]) for row in summary])
    mean = np.asarray([float(row["mean_exploitability"]) for row in summary])
    se = np.asarray([float(row["se_exploitability"]) for row in summary])
    maximum_seed_count = max(int(row["n_seeds"]) for row in summary)
    complete = np.asarray([
        int(row["n_seeds"]) == maximum_seed_count for row in summary
    ])
    mean = np.where(complete, mean, np.nan)
    se = np.where(complete, se, np.nan)
    ax.plot(
        x, mean, color="#174A73", linewidth=2.2,
        drawstyle="steps-post", label="Cross-seed mean"
    )
    ax.fill_between(
        x, mean - se, mean + se, color="#2878B5", alpha=0.20,
        step="post", label="±1 SE"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exploitability (NashConv / 2)")
    experiment_id = int(rows[0]["experiment_id"])
    set_chart_title(ax, f"Experiment {experiment_id}: source-policy exploitability")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def export_aggregate_trajectory(output_dir: Path, rows: Sequence[Mapping]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_trajectory_rows(output_dir / "source_trajectory.csv", rows)
    summary = write_trajectory_summary(
        output_dir / "source_trajectory_summary.csv", rows
    )
    plot_trajectory(
        rows,
        summary,
        x_field="training_hours",
        summary_x_field="mean_training_hours",
        xlabel="Training time (hours)",
        output=output_dir / "source_exploitability_by_training_time.png",
    )
    plot_trajectory(
        rows,
        summary,
        x_field="nodes_touched",
        summary_x_field="mean_nodes_touched",
        xlabel="Nodes touched",
        output=output_dir / "source_exploitability_by_nodes.png",
    )


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "TRAJECTORY_FIELDS",
    "build_final_policy_row",
    "build_trajectory_rows",
    "export_aggregate_trajectory",
    "plot_trajectory",
    "read_trajectory_rows",
    "summarise_trajectory_rows",
    "write_trajectory_rows",
    "write_trajectory_summary",
]
