"""CLI for Experiment 40's sequential/parallel ESCHER comparison."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import (
    DEFAULT_PAIRED_DELTA_FIELDS,
    run_variant_ablation,
)

from .config import (
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    VARIANTS,
)


RUNTIME_FIELDS = [
    "solver_initialization_seconds",
    "elapsed_seconds",
    "end_to_end_seconds",
    "regret_experience_collection_seconds",
    "value_experience_collection_seconds",
    "experience_collection_seconds",
]

PAIRED_DELTA_FIELDS = (
    list(DEFAULT_PAIRED_DELTA_FIELDS)
    + ["final_policy_value", "final_nodes_touched"]
    + RUNTIME_FIELDS
)

PAIRED_RATIO_FIELDS = {
    "solver_initialization_speedup": "solver_initialization_seconds",
    "training_loop_speedup": "elapsed_seconds",
    "end_to_end_speedup": "end_to_end_seconds",
    "regret_collection_speedup": "regret_experience_collection_seconds",
    "value_collection_speedup": "value_experience_collection_seconds",
    "experience_collection_speedup": "experience_collection_seconds",
}

EQUIVALENCE_TOLERANCES = {
    "final_exploitability": FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    "final_policy_value": FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
}

TIMING_CURVE_PLOT_SPECS = [
    (
        "cumulative_experience_collection_seconds",
        "Cumulative experience-collection time (seconds)",
        "ESCHER sequential versus parallel: collection time by nodes touched",
        "experience_collection_seconds_by_nodes.png",
        None,
        None,
    ),
]


def main(argv: Optional[List[str]] = None) -> int:
    return run_variant_ablation(
        argv,
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        variants=VARIANTS,
        baseline_variant_id=BASELINE_VARIANT_ID,
        output_root="outputs/parallel_equivalence_ablation",
        description=(
            "Compare sequential and Ray-parallel Experiment 28 ESCHER over "
            "three paired seeds."
        ),
        logger_name="escher_poker.experiment.parallel_equivalence_ablation",
        progress_label="Sequential/parallel variants",
        plot_title_prefix="ESCHER sequential versus Ray-parallel",
        worker_module=(
            "experiments.leduc_poker.escher_parallel_equivalence_ablation.run"
        ),
        paired_delta_fields=PAIRED_DELTA_FIELDS,
        paired_ratio_fields=PAIRED_RATIO_FIELDS,
        paired_equivalence_tolerances=EQUIVALENCE_TOLERANCES,
        extra_curve_plot_specs=TIMING_CURVE_PLOT_SPECS,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
