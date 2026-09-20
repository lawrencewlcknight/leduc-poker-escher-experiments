"""Contract, scheduling and smoke tests for Experiment 49."""

import json
import math
import subprocess
import sys
from pathlib import Path

from experiments.leduc_poker.escher_paper_36h_trajectory.cloud import (
    SMOKE_SEEDS,
    aggregate_workers,
    run_evaluation_worker,
    run_training_worker,
    task_name,
)
from experiments.leduc_poker.escher_paper_36h_trajectory.config import (
    CHECKPOINT_INTERVAL_SECONDS,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    PAPER_HYPERPARAMETERS,
    SELECTED_POLICY_ARM,
    TARGET_NODES_TOUCHED,
    TRAINING_WALL_CLOCK_SECONDS,
)
from experiments.leduc_poker.escher_paper_36h_trajectory.run import build_config


def test_experiment_49_frozen_scientific_contract():
    assert EXPERIMENT_ID == 49
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert TRAINING_WALL_CLOCK_SECONDS == 36 * 60 * 60
    assert CHECKPOINT_INTERVAL_SECONDS == 30 * 60
    assert TARGET_NODES_TOUCHED == 15_000_000
    assert DEFAULT_CONFIG["training_wall_clock_seconds"] == 36 * 60 * 60
    assert DEFAULT_CONFIG["checkpoint_interval_seconds"] == 30 * 60
    assert DEFAULT_CONFIG["average_policy_memory_capacity"] == 1_000_000
    assert DEFAULT_CONFIG["selected_policy_arm"] == SELECTED_POLICY_ARM
    assert DEFAULT_CONFIG["selected_policy_grouped"] is True
    assert DEFAULT_CONFIG["compute_exploitability"] is False
    assert DEFAULT_CONFIG["fit_final_policy_after_training"] is False
    assert DEFAULT_CONFIG["exclude_checkpoint_time_from_training_budget"] is True
    for key, expected in PAPER_HYPERPARAMETERS.items():
        assert DEFAULT_CONFIG[key] == expected


def test_smoke_contract_reduces_all_expensive_dimensions():
    config = build_config(smoke=True)
    assert config["num_iterations"] == 2
    assert config["training_wall_clock_seconds"] == 1e-9
    assert config["target_nodes_touched"] == 1
    assert config["average_policy_memory_capacity"] == 256
    assert config["policy_network_train_steps"] == 2
    assert config["regret_network_train_steps"] == 1
    assert config["value_network_train_steps"] == 1


def test_task_schedule_has_one_seed_per_task():
    assert [task_name(index, DEFAULT_SEEDS) for index in range(5)] == [
        "task_000_seed_1234",
        "task_001_seed_2025",
        "task_002_seed_31415",
        "task_003_seed_27182",
        "task_004_seed_16180",
    ]


def _build_job(tmp_path: Path, kind: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / f"{kind}.json"
    command = [
        sys.executable,
        str(root / "gcp" / "escher_paper_36h_trajectory_batch.py"),
        "--kind",
        kind,
        "--output",
        str(output),
        "--run-id",
        "exp49-test",
        "--bucket-root",
        "gs://test-bucket",
        "--service-account",
        "test@example.invalid",
        "--repo-ref",
        "deadbeef",
        "--parallelism",
        "5",
    ]
    if kind == "controller":
        command.extend(["--project-id", "test-project", "--region", "europe-west1"])
    subprocess.run(command, check=True)
    with open(output, encoding="utf-8") as handle:
        return json.load(handle)


def test_training_and_evaluation_use_five_separate_parallel_vms(tmp_path):
    for kind, duration in (("train", "172800s"), ("evaluate", "72000s")):
        job = _build_job(tmp_path, kind)
        group = job["taskGroups"][0]
        assert group["taskCount"] == 5
        assert group["parallelism"] == 5
        assert group["taskCountPerNode"] == 1
        assert group["taskSpec"]["maxRunDuration"] == duration
        assert group["taskSpec"]["maxRetryCount"] == 0
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"


def test_controller_runs_training_then_evaluation_then_aggregation(tmp_path):
    job = _build_job(tmp_path, "controller")
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "run_escher_paper_36h_trajectory.sh" in script
    assert "EXP49_REMOTE_CONTROLLER=1" in script


def test_end_to_end_smoke_produces_temporal_and_endpoint_artifacts(tmp_path):
    run_training_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    run_evaluation_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        training_root=tmp_path,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    result = aggregate_workers(
        evaluation_root=tmp_path,
        seeds=SMOKE_SEEDS,
        output_dir=tmp_path / "analysis",
        smoke=True,
    )
    assert result["status"] == "complete"
    assert result["num_seeds"] == 1
    assert (tmp_path / "analysis" / "exploitability_by_training_time.png").is_file()
    assert (tmp_path / "analysis" / "exploitability_by_nodes.png").is_file()
    assert (tmp_path / "analysis" / "distillation_gap_by_training_time.png").is_file()
    assert (tmp_path / "analysis" / "trajectory_by_nodes_summary.csv").is_file()
    assert (tmp_path / "analysis" / "endpoint_15m_seed_metrics.csv").is_file()
    assert (tmp_path / "analysis" / "endpoint_36h_seed_metrics.csv").is_file()
    assert (tmp_path / "analysis" / "thesis_15m_aggregate_summary.csv").is_file()

    training_result = json.loads(
        next((tmp_path / "training").rglob("training_result.json")).read_text(encoding="utf-8")
    )
    assert training_result["checkpoint_time_excluded_seconds"] > 0.0
    assert training_result["hit_wall_clock_limit"] is True
    assert math.isfinite(training_result["active_training_seconds"])
