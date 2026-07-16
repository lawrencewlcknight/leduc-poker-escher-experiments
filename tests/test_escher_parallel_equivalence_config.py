"""Configuration checks for Experiment 40 parallel equivalence."""

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
)
from experiments.leduc_poker.escher_parallel_equivalence_ablation.config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    PARALLEL_NUM_WORKERS,
    PARALLEL_VARIANT_ID,
    VARIANTS,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config


def test_parallel_equivalence_uses_three_paired_seeds():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]


def test_parallel_arm_changes_only_execution_metadata_from_experiment_28():
    sequential = make_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    parallel = make_variant_config(DEFAULT_CONFIG, VARIANTS[1])

    ignored = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "baseline_variant_id",
        "ablation_variants",
        "execution_backend",
        "parallel_num_workers",
        "parallel_ray_address",
        "parallel_log_to_driver",
        "save_final_checkpoints",
    }
    for key, value in deepcopy(EXPERIMENT_28_CONFIG).items():
        if key not in ignored:
            assert sequential[key] == value
            assert parallel[key] == value

    assert sequential["variant_id"] == BASELINE_VARIANT_ID
    assert sequential["execution_backend"] == "sequential"
    assert sequential["parallel_num_workers"] == 1
    assert parallel["variant_id"] == PARALLEL_VARIANT_ID
    assert parallel["execution_backend"] == "ray_parallel"
    assert parallel["parallel_num_workers"] == PARALLEL_NUM_WORKERS == 3


def test_equivalence_margins_are_positive_and_predeclared():
    assert FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN == 0.05
    assert FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN == 0.02
