"""CLI for Experiment 41's combined ESCHER candidate ablation."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import (
    DEFAULT_PAIRED_DELTA_FIELDS,
    run_variant_ablation,
)

from .config import (
    BASELINE_VARIANT_ID,
    CANDIDATE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)


REPLAY_CURVE_PLOT_SPECS = [
    (
        "regret_replay_samples_per_infoset_cv_player_0",
        "CV of stored samples per player-0 infoset",
        "ESCHER combined candidate: replay imbalance by nodes touched",
        "regret_replay_infoset_cv_player_0_by_nodes.png",
        None,
        None,
    ),
    (
        "regret_replay_retention_fraction_player_0",
        "Retained fraction of player-0 regret stream",
        "ESCHER combined candidate: replay retention by nodes touched",
        "regret_replay_retention_player_0_by_nodes.png",
        None,
        None,
    ),
]

PAIRED_DELTA_FIELDS = list(DEFAULT_PAIRED_DELTA_FIELDS) + [
    "intermediate_final_window_mean_exploitability",
]


def main(argv: Optional[List[str]] = None) -> int:
    return run_variant_ablation(
        argv,
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        variants=VARIANTS,
        baseline_variant_id=BASELINE_VARIANT_ID,
        output_root="outputs/combined_candidate_ablation",
        description=(
            "Compare Experiment 28 with the combined ESCHER candidate and "
            "the exact-balanced maximum stack."
        ),
        logger_name="escher_poker.experiment.combined_candidate_ablation",
        progress_label="Combined candidate variants",
        plot_title_prefix="ESCHER combined candidate ablation",
        worker_module=(
            "experiments.leduc_poker.escher_combined_candidate_ablation.run"
        ),
        paired_delta_fields=PAIRED_DELTA_FIELDS,
        additional_paired_baseline_ids=[CANDIDATE_VARIANT_ID],
        extra_curve_plot_specs=REPLAY_CURVE_PLOT_SPECS,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
