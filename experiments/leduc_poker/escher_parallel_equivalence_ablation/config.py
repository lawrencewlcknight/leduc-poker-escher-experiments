"""Configuration for Experiment 40's parallel-equivalence ablation."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
BASELINE_VARIANT_ID = "experiment_28_sequential"
PARALLEL_VARIANT_ID = "experiment_28_ray_parallel"
PARALLEL_NUM_WORKERS = 3

# Pre-declared practical-equivalence margins. These are absolute differences
# in final metrics, not tuning objectives selected after observing results.
FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN = 0.05
FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN = 0.02


VARIANTS = [
    {
        "variant_id": BASELINE_VARIANT_ID,
        "variant_label": "Experiment 28 sequential",
        "variant_description": (
            "Unmodified Experiment 28 learner and sequential experience collection."
        ),
        "execution_backend": "sequential",
        "parallel_num_workers": 1,
    },
    {
        "variant_id": PARALLEL_VARIANT_ID,
        "variant_label": "Experiment 28 Ray parallel (3 workers)",
        "variant_description": (
            "Experiment 28 learner with traversal collection partitioned over "
            "three Ray actors and one central learner."
        ),
        "execution_backend": "ray_parallel",
        "parallel_num_workers": PARALLEL_NUM_WORKERS,
    },
]


DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_parallel_equivalence_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "execution_backend": "sequential",
    "parallel_num_workers": PARALLEL_NUM_WORKERS,
    "parallel_ray_address": None,
    "parallel_log_to_driver": False,
    "save_final_checkpoints": False,
})
