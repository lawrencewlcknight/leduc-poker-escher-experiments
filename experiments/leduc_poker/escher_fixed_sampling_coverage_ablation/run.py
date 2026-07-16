"""CLI for Experiment 39 ESCHER fixed sampling-policy coverage ablation."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import (
    run_variant_ablation,
)

from .config import BASELINE_VARIANT_ID, DEFAULT_CONFIG, DEFAULT_SEEDS, VARIANTS


SAMPLING_CURVE_PLOT_SPECS = [
    (
        "fixed_sampling_own_history_reach_min_player_0",
        "Minimum fixed own-policy reach (player 0)",
        "ESCHER fixed sampling: exact minimum history reach",
        "fixed_sampling_minimum_history_reach_by_nodes.png",
        None,
        None,
    ),
    (
        "sampling_coverage_unique_infosets_player_0",
        "Observed player-0 infosets",
        "ESCHER fixed sampling: cumulative infoset coverage",
        "sampling_unique_infosets_by_nodes.png",
        None,
        None,
    ),
    (
        "sampling_coverage_visits_cv_player_0",
        "CV of visits per player-0 infoset",
        "ESCHER fixed sampling: empirical visit imbalance",
        "sampling_infoset_visit_cv_by_nodes.png",
        None,
        None,
    ),
    (
        "sampling_coverage_observed_own_reach_min_player_0",
        "Minimum observed player-0 sampling reach",
        "ESCHER fixed sampling: observed minimum own-policy reach",
        "sampling_observed_minimum_reach_by_nodes.png",
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
        output_root="outputs/fixed_sampling_coverage_ablation",
        description="Run the Leduc poker ESCHER fixed sampling-policy ablation.",
        logger_name="escher_poker.experiment.fixed_sampling_coverage_ablation",
        progress_label="Fixed sampling variants",
        plot_title_prefix="ESCHER fixed sampling-policy coverage ablation",
        worker_module=(
            "experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run"
        ),
        extra_curve_plot_specs=SAMPLING_CURVE_PLOT_SPECS,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
