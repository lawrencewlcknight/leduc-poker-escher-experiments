"""CLI for Experiment 35 ESCHER regret-target specification ablation."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import (
    DEFAULT_PAIRED_DELTA_FIELDS,
    run_variant_ablation,
)

from .config import BASELINE_VARIANT_ID, DEFAULT_CONFIG, DEFAULT_SEEDS, VARIANTS


TARGET_DIAGNOSTIC_DELTA_FIELDS = [
    "last_intermediate_regret_target_bellman_residual_abs_mean_player_0",
    "last_intermediate_regret_target_bellman_residual_abs_mean_player_1",
    "last_intermediate_regret_target_policy_weighted_target_abs_mean_player_0",
    "last_intermediate_regret_target_policy_weighted_target_abs_mean_player_1",
    "last_intermediate_regret_target_all_legal_targets_negative_fraction_player_0",
    "last_intermediate_regret_target_all_legal_targets_negative_fraction_player_1",
]


def main(argv: Optional[List[str]] = None) -> int:
    return run_variant_ablation(
        argv,
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        variants=VARIANTS,
        baseline_variant_id=BASELINE_VARIANT_ID,
        output_root="outputs/regret_target_specification_ablation",
        description="Run the Leduc poker ESCHER regret-target specification ablation.",
        logger_name="escher_poker.experiment.regret_target_specification_ablation",
        progress_label="Regret-target variants",
        plot_title_prefix="ESCHER regret-target specification ablation",
        worker_module=(
            "experiments.leduc_poker.escher_regret_target_specification_ablation.run"
        ),
        paired_delta_fields=(
            list(DEFAULT_PAIRED_DELTA_FIELDS) + TARGET_DIAGNOSTIC_DELTA_FIELDS
        ),
        extra_summary_fields={
            "regret_target_baseline_variant": "regret_target_baseline",
        },
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
