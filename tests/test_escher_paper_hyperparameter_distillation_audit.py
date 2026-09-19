"""Contract and cloud-scheduling tests for Experiment 46."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    DEFAULT_CONFIG as EXPERIMENT_45_CONFIG,
)
from experiments.leduc_poker.escher_paper_hyperparameter_distillation_audit.config import (
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PAPER_HYPERPARAMETERS,
    TRAINING_WALL_CLOCK_SECONDS,
)
from experiments.leduc_poker.escher_paper_hyperparameter_distillation_audit.run import (
    build_config,
)


def _smoke_args():
    return Namespace(
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


def test_experiment_46_changes_only_named_scientific_parameters():
    changed = {
        key for key in DEFAULT_CONFIG
        if DEFAULT_CONFIG.get(key) != EXPERIMENT_45_CONFIG.get(key)
    }
    assert changed == {"experiment_name", *PAPER_HYPERPARAMETERS}
    assert EXPERIMENT_ID == 46
    assert DEFAULT_CONFIG["experiment_name"] == EXPERIMENT_NAME
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert DEFAULT_CONFIG["training_wall_clock_seconds"] == 12 * 60 * 60
    assert TRAINING_WALL_CLOCK_SECONDS == 12 * 60 * 60
    assert DEFAULT_CONFIG["average_policy_memory_capacity"] == 1_000_000
    assert DEFAULT_CONFIG["memory_capacity"] == 50_000


def test_experiment_46_exact_paper_hyperparameters():
    assert PAPER_HYPERPARAMETERS == {
        "num_traversals": 1_000,
        "num_val_fn_traversals": 1_000,
        "batch_size_regret": 2_048,
        "batch_size_value": 2_048,
        "regret_network_train_steps": 5_000,
        "value_network_train_steps": 5_000,
        "policy_network_train_steps": 10_000,
    }
    for key, expected in PAPER_HYPERPARAMETERS.items():
        assert DEFAULT_CONFIG[key] == expected


def test_experiment_46_smoke_reduces_expensive_paper_settings():
    config = build_config(_smoke_args())
    assert config["experiment_name"] == EXPERIMENT_NAME
    assert config["num_iterations"] == 2
    assert config["num_traversals"] == 2
    assert config["num_val_fn_traversals"] == 2
    assert config["batch_size_regret"] == 2
    assert config["batch_size_value"] == 2
    assert config["regret_network_train_steps"] == 1
    assert config["value_network_train_steps"] == 1
    assert config["policy_network_train_steps"] == 2


def test_experiment_46_batch_builder_uses_three_parallel_vms(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    builder = (
        repository_root
        / "gcp"
        / "escher_paper_hyperparameter_distillation_audit_batch.py"
    )
    output = tmp_path / "train.json"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--kind", "train",
            "--output", str(output),
            "--run-id", "exp46-test",
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
    assert task_group["taskSpec"]["maxRunDuration"] == "172800s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    script = task_group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "escher_paper_hyperparameter_distillation_audit.cloud" in script

