"""Contract and mathematical tests for Experiment 45."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyspiel

from escher_poker.experiment_utils import make_escher_solver
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    ARM_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    SAFETY_MAX_ITERATIONS,
    TRAINING_WALL_CLOCK_SECONDS,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.cloud import (
    task_name,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.distillation import (
    FrozenReservoir,
    cross_entropy_objective_equivalence,
    group_reservoir,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.run import (
    build_config,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.compare_trajectories import (
    compare_trajectories,
)
from experiments.leduc_poker.escher_frozen_policy_distillation_audit.trajectory import (
    build_final_policy_row,
    build_trajectory_rows,
    summarise_trajectory_rows,
    write_trajectory_rows,
)


def test_experiment_45_preserves_core_memories_and_enlarges_only_policy():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert DEFAULT_CONFIG["num_iterations"] == SAFETY_MAX_ITERATIONS
    assert DEFAULT_CONFIG["training_wall_clock_seconds"] == 12 * 60 * 60
    assert TRAINING_WALL_CLOCK_SECONDS == 12 * 60 * 60
    assert DEFAULT_CONFIG["expected_final_nodes_touched"] is None
    assert DEFAULT_CONFIG["check_exploitability_every"] == 10
    assert DEFAULT_CONFIG["compute_exploitability"] is False
    assert DEFAULT_CONFIG["training_progress_every"] == 10
    assert DEFAULT_CONFIG["source_policy_fit_mode"] == (
        "final_only_after_timed_training"
    )
    assert DEFAULT_CONFIG["memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["regret_memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["value_memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["value_validation_memory_capacity"] == 50_000
    assert DEFAULT_CONFIG["average_policy_memory_capacity"] == 1_000_000
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 256, 128)
    assert tuple(DEFAULT_CONFIG["distillation_arms"]) == ARM_ORDER


def test_grouped_cross_entropy_is_exact_row_objective():
    reservoir = FrozenReservoir(
        info_states=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        action_probs=np.asarray(
            [[0.8, 0.2], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32
        ),
        iterations=np.asarray([1, 3, 2], dtype=np.float32),
        legal_actions=np.ones((3, 2), dtype=np.float32),
        reach_probs=np.ones(3, dtype=np.float32),
        obs_indices=np.arange(3, dtype=np.float32),
    )
    grouped = group_reservoir(reservoir, iteration=4)
    predictions = np.asarray([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
    row, compressed = cross_entropy_objective_equivalence(
        reservoir, grouped, predictions, iteration=4
    )
    assert np.isclose(row, compressed, atol=1e-7)


def _trajectory_rows(experiment_id, seed, length=3):
    diagnostics = {
        "iteration": np.arange(1, length + 1) * 10,
        "solver_iteration": np.arange(1, length + 1) * 10,
        "wall_clock_seconds": np.arange(1, length + 1) * 900.0,
    }
    for name in (
        "policy_loss",
        "value_loss",
        "value_test_loss",
        "regret_loss_player_0",
        "regret_loss_player_1",
        "peak_rss_mb",
        "cumulative_experience_collection_seconds",
    ):
        diagnostics[name] = np.arange(length, dtype=float)
    for name in (
        "average_policy_buffer_size",
        "regret_buffer_size_player_0",
        "regret_buffer_size_player_1",
        "value_buffer_size",
        "value_test_buffer_size",
    ):
        diagnostics[name] = np.arange(length, dtype=int)
    return build_trajectory_rows(
        experiment_id=experiment_id,
        experiment_name=f"experiment_{experiment_id}",
        seed=seed,
        nash_convs=np.linspace(0.4, 0.2, length),
        nodes_touched=np.arange(1, length + 1) * 1_000,
        average_policy_values=np.linspace(-0.1, -0.08, length),
        diagnostics=diagnostics,
    )


def test_trajectory_schema_records_time_nodes_and_exploitability():
    rows = _trajectory_rows(45, 1234)
    assert len(rows) == 3
    assert rows[0]["training_hours"] == 0.25
    assert rows[0]["nodes_touched"] == 1_000
    assert rows[0]["exploitability"] == 0.2
    assert rows[-1]["checkpoint_index"] == 2


def test_trajectory_summary_accepts_variable_seed_lengths():
    rows = _trajectory_rows(45, 1234, length=3)
    rows.extend(_trajectory_rows(45, 2025, length=2))
    summary = summarise_trajectory_rows(rows)
    assert len(summary) == 3
    assert summary[0]["n_seeds"] == 2
    assert summary[-1]["n_seeds"] == 1


def test_final_policy_rows_are_aggregated_separately_from_variable_checkpoints():
    rows = _trajectory_rows(45, 1234, length=3)
    rows.extend(_trajectory_rows(45, 2025, length=2))
    for seed, checkpoint_index in ((1234, 3), (2025, 2)):
        rows.append(build_final_policy_row(
            experiment_id=45,
            experiment_name="experiment_45",
            seed=seed,
            checkpoint_index=checkpoint_index,
            iteration=30,
            nodes_touched=4_000,
            wall_clock_seconds=3_600,
            metrics={"nash_conv": 0.1, "exploitability": 0.05, "policy_value": -0.08},
            final_policy_loss=0.01,
            diagnostics={},
        ))
    summary = summarise_trajectory_rows(rows)
    final_rows = [row for row in summary if row["is_final_policy_fit"]]
    assert len(final_rows) == 1
    assert final_rows[0]["checkpoint_index"] == -1
    assert final_rows[0]["n_seeds"] == 2
    assert final_rows[0]["mean_exploitability"] == 0.05


def test_cross_experiment_comparison_writes_temporal_charts(tmp_path):
    inputs = []
    for experiment_id in (45, 46):
        analysis = tmp_path / f"exp{experiment_id}" / "analysis"
        trajectory = analysis / "source_trajectory.csv"
        rows = _trajectory_rows(experiment_id, 1234)
        rows.extend(_trajectory_rows(experiment_id, 2025))
        write_trajectory_rows(trajectory, rows)
        inputs.append((f"Experiment {experiment_id}", trajectory))
    output = tmp_path / "comparison"
    result = compare_trajectories(
        inputs, output, time_step_hours=0.25, node_grid_points=5
    )
    assert result["num_raw_rows"] == 12
    assert (output / "combined_source_trajectory.csv").is_file()
    assert (output / "combined_trajectory_summary.csv").is_file()
    assert (output / "combined_exploitability_by_training_time.png").is_file()
    assert (output / "combined_exploitability_by_nodes.png").is_file()


def test_smoke_config_reduces_all_expensive_dimensions():
    args = Namespace(
        iterations=None,
        traversals=None,
        value_traversals=None,
        evaluation_interval=None,
        memory_capacity=None,
        average_policy_memory_capacity=None,
        policy_network_train_steps=None,
        regret_network_train_steps=None,
        value_network_train_steps=None,
        batch_size_regret=None,
        batch_size_value=None,
        batch_size_average_policy=None,
        policy_network_layers=None,
        regret_network_layers=None,
        value_network_layers=None,
        reservoir_decode_chunk_size=None,
        smoke=True,
    )
    config = build_config(args)
    assert config["num_iterations"] == 2
    assert config["average_policy_memory_capacity"] == 256
    assert config["policy_network_layers"] == (8, 8)
    assert config["policy_network_train_steps"] == 2
    assert config["compute_exploitability"] is False
    assert config["training_progress_every"] == 1


def test_solver_uses_independent_memory_capacities():
    config = dict(DEFAULT_CONFIG)
    config.update({
        "num_iterations": 0,
        "memory_capacity": 7,
        "regret_memory_capacity": 11,
        "value_memory_capacity": 13,
        "value_validation_memory_capacity": 17,
        "average_policy_memory_capacity": 19,
        "policy_network_layers": (8, 8),
        "regret_network_layers": (8, 8),
        "value_network_layers": (8, 8),
        "regret_network_head_units": 4,
    })
    solver = make_escher_solver(pyspiel.load_game("leduc_poker"), config)
    assert solver._average_policy_memories._reservoir_buffer_capacity == 19
    assert solver._regret_memories[0]._reservoir_buffer_capacity == 11
    assert solver._regret_memories[1]._reservoir_buffer_capacity == 11
    assert solver._value_memory._reservoir_buffer_capacity == 13
    assert solver._value_memory_test._reservoir_buffer_capacity == 17
    solver._value_memory.add(b"train")
    solver._value_memory_test.add(b"validation")
    solver.clear_val_memories()
    solver.clear_val_memories_test()
    assert solver.get_value_memory_peak_counts() == {
        "training": 1,
        "validation": 1,
    }


def test_solver_wall_clock_budget_stops_at_a_completed_iteration():
    args = Namespace(
        iterations=100,
        traversals=None,
        value_traversals=None,
        evaluation_interval=None,
        memory_capacity=None,
        average_policy_memory_capacity=None,
        policy_network_train_steps=None,
        regret_network_train_steps=None,
        value_network_train_steps=None,
        batch_size_regret=None,
        batch_size_value=None,
        batch_size_average_policy=None,
        policy_network_layers=None,
        regret_network_layers=None,
        value_network_layers=None,
        reservoir_decode_chunk_size=None,
        smoke=True,
    )
    config = build_config(args)
    solver = make_escher_solver(pyspiel.load_game("leduc_poker"), config)
    progress = []
    _, _, convs, nodes, values, _ = solver.solve(
        max_wall_clock_seconds=1e-9,
        post_iteration_callback=lambda _solver, row: progress.append(row),
    )
    summary = solver.get_last_solve_summary()
    assert summary["termination_reason"] == "wall_clock_limit"
    assert summary["hit_wall_clock_limit"] is True
    assert summary["completed_solve_passes"] == 1
    assert summary["active_training_seconds"] >= 1e-9
    assert summary["nodes_touched"] == solver.get_num_nodes()
    assert progress
    assert not convs and not nodes and not values


def test_cloud_task_schedule_is_one_task_per_production_seed():
    assert [task_name(index, DEFAULT_SEEDS) for index in range(3)] == [
        "task_000_seed_1234",
        "task_001_seed_2025",
        "task_002_seed_31415",
    ]


def test_batch_builder_runs_all_seed_tasks_on_separate_parallel_vms(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    builder = repository_root / "gcp" / "escher_frozen_policy_distillation_audit_batch.py"
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--kind", "train",
            "--output", str(output),
            "--run-id", "exp45-test",
            "--bucket-root", "gs://test-bucket",
            "--service-account", "test@example.invalid",
            "--repo-ref", "deadbeef",
        ],
        check=True,
    )
    with open(output, encoding="utf-8") as handle:
        job = json.load(handle)
    task_group = job["taskGroups"][0]
    assert task_group["taskCount"] == 3
    assert task_group["parallelism"] == 3
    assert task_group["taskCountPerNode"] == 1
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    script = task_group["taskSpec"]["runnables"][0]["script"]["text"]
    assert 'git -C "$REPOSITORY" checkout --detach "$REPO_REF"' in script
    assert "escher_frozen_policy_distillation_audit.cloud" in script
    assert "ESCHER_AUDIT_TASK_METADATA" in script
    assert "json.load(sys.stdin)" not in script


def test_controller_fails_fast_when_child_job_listing_is_forbidden(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    builder = repository_root / "gcp" / "escher_frozen_policy_distillation_audit_batch.py"
    output = tmp_path / "controller.json"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--kind", "controller",
            "--output", str(output),
            "--run-id", "exp45-test",
            "--bucket-root", "gs://test-bucket",
            "--service-account", "test@example.invalid",
            "--repo-ref", "deadbeef",
            "--project-id", "test-project",
            "--region", "europe-west1",
        ],
        check=True,
    )
    with open(output, encoding="utf-8") as handle:
        job = json.load(handle)
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "gcloud batch jobs list" in script
    assert "--limit 1" in script
