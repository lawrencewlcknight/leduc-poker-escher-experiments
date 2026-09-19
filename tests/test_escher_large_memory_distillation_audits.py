"""Contracts and cloud scheduling for Experiments 47 and 48."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

from experiments.leduc_poker.escher_frozen_policy_distillation_audit.config import (
    DEFAULT_CONFIG as EXPERIMENT_45_CONFIG,
)
from experiments.leduc_poker.escher_large_regret_reservoir_distillation_audit.config import (
    DEFAULT_CONFIG as EXPERIMENT_47_CONFIG,
    EXPERIMENT_ID as EXPERIMENT_47_ID,
)
from experiments.leduc_poker.escher_large_regret_reservoir_distillation_audit.run import (
    build_config as build_experiment_47_config,
)
from experiments.leduc_poker.escher_all_large_memory_distillation_audit.config import (
    DEFAULT_CONFIG as EXPERIMENT_48_CONFIG,
    EXPERIMENT_ID as EXPERIMENT_48_ID,
)
from experiments.leduc_poker.escher_all_large_memory_distillation_audit.run import (
    build_config as build_experiment_48_config,
)


def _smoke_args():
    return Namespace(
        iterations=None,
        traversals=None,
        value_traversals=None,
        evaluation_interval=None,
        memory_capacity=None,
        regret_memory_capacity=None,
        value_memory_capacity=None,
        value_validation_memory_capacity=None,
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


def test_experiment_47_changes_only_regret_capacity_and_name():
    changed = {
        key for key in EXPERIMENT_47_CONFIG
        if EXPERIMENT_47_CONFIG.get(key) != EXPERIMENT_45_CONFIG.get(key)
    }
    assert changed == {"experiment_name", "regret_memory_capacity"}
    assert EXPERIMENT_47_ID == 47
    assert EXPERIMENT_47_CONFIG["regret_memory_capacity"] == 1_000_000
    assert EXPERIMENT_47_CONFIG["value_memory_capacity"] == 50_000
    assert EXPERIMENT_47_CONFIG["value_validation_memory_capacity"] == 50_000
    assert EXPERIMENT_47_CONFIG["average_policy_memory_capacity"] == 1_000_000


def test_experiment_48_changes_only_value_capacities_from_experiment_47():
    changed = {
        key for key in EXPERIMENT_48_CONFIG
        if EXPERIMENT_48_CONFIG.get(key) != EXPERIMENT_47_CONFIG.get(key)
    }
    assert changed == {
        "experiment_name",
        "memory_capacity",
        "value_memory_capacity",
        "value_validation_memory_capacity",
    }
    assert EXPERIMENT_48_ID == 48
    for key in (
        "memory_capacity",
        "regret_memory_capacity",
        "value_memory_capacity",
        "value_validation_memory_capacity",
        "average_policy_memory_capacity",
    ):
        assert EXPERIMENT_48_CONFIG[key] == 1_000_000


def test_large_memory_smoke_contracts_reduce_all_capacities():
    for build_config in (
        build_experiment_47_config,
        build_experiment_48_config,
    ):
        config = build_config(_smoke_args())
        assert config["regret_memory_capacity"] == 128
        assert config["value_memory_capacity"] == 128
        assert config["value_validation_memory_capacity"] == 128
        assert config["average_policy_memory_capacity"] == 256


def test_large_memory_batch_builders_use_three_parallel_vms(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    cases = (
        (
            "escher_large_regret_reservoir_distillation_audit_batch.py",
            "escher_large_regret_reservoir_distillation_audit.cloud",
        ),
        (
            "escher_all_large_memory_distillation_audit_batch.py",
            "escher_all_large_memory_distillation_audit.cloud",
        ),
    )
    for builder_name, module_name in cases:
        output = tmp_path / f"{builder_name}.json"
        subprocess.run(
            [
                sys.executable,
                str(repository_root / "gcp" / builder_name),
                "--kind", "train",
                "--output", str(output),
                "--run-id", "memory-audit-test",
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
        assert task_group["taskSpec"]["maxRunDuration"] == "86400s"
        assert (
            job["allocationPolicy"]["instances"][0]["policy"]["machineType"]
            == "n2-standard-8"
        )
        script = task_group["taskSpec"]["runnables"][0]["script"]["text"]
        assert module_name in script
