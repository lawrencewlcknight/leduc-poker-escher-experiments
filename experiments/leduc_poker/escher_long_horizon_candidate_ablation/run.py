"""CLI for Experiment 42's approximately 10x-node ESCHER ablation."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import (
    DEFAULT_PAIRED_DELTA_FIELDS,
    run_variant_ablation,
)

from .config import BASELINE_VARIANT_ID, DEFAULT_CONFIG, DEFAULT_SEEDS, VARIANTS


PAIRED_DELTA_FIELDS = list(DEFAULT_PAIRED_DELTA_FIELDS) + [
    "intermediate_final_window_mean_exploitability",
    "elapsed_seconds",
]

REPLAY_CURVE_PLOT_SPECS = [
    (
        "regret_replay_samples_per_infoset_cv_player_0",
        "CV of stored samples per player-0 infoset",
        "ESCHER 10x-node candidate: replay imbalance by nodes touched",
        "regret_replay_infoset_cv_player_0_by_nodes.png",
        None,
        None,
    ),
    (
        "regret_replay_retention_fraction_player_0",
        "Retained fraction of player-0 regret stream",
        "ESCHER 10x-node candidate: replay retention by nodes touched",
        "regret_replay_retention_player_0_by_nodes.png",
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
        output_root="outputs/long_horizon_candidate_ablation",
        description=(
            "Compare the Experiment 28 baseline and policy-Q stratified "
            "candidate over an approximately 10x solve-pass budget."
        ),
        logger_name="escher_poker.experiment.long_horizon_candidate_ablation",
        progress_label="10x-node candidate variants",
        plot_title_prefix="ESCHER 10x-node candidate ablation",
        worker_module=(
            "experiments.leduc_poker.escher_long_horizon_candidate_ablation.run"
        ),
        paired_delta_fields=PAIRED_DELTA_FIELDS,
        extra_summary_fields={
            "target_solve_pass_multiplier": "target_solve_pass_multiplier",
            "solve_pass_multiplier": "solve_pass_multiplier",
            "experiment_28_solve_passes": "experiment_28_solve_passes",
            "long_run_solve_passes": "long_run_solve_passes",
        },
        extra_curve_plot_specs=REPLAY_CURVE_PLOT_SPECS,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
