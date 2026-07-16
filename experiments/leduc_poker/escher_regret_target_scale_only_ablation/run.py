"""CLI for Experiment 36 corrected regret-target processing ablation."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import (
    DEFAULT_PAIRED_DELTA_FIELDS,
    run_variant_ablation,
)

from .config import (
    BASELINE_VARIANT_ID,
    CORRECTED_STANDARDIZED_CONTROL_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    VARIANTS,
)


PROCESSING_DIAGNOSTIC_DELTA_FIELDS = [
    "last_intermediate_regret_target_sign_flip_fraction_player_0",
    "last_intermediate_regret_target_sign_flip_fraction_player_1",
]


def main(argv: Optional[List[str]] = None) -> int:
    return run_variant_ablation(
        argv,
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        variants=VARIANTS,
        baseline_variant_id=BASELINE_VARIANT_ID,
        output_root="outputs/regret_target_scale_only_ablation",
        description=(
            "Run the Leduc poker ESCHER corrected regret-target scale-only ablation."
        ),
        logger_name="escher_poker.experiment.regret_target_scale_only_ablation",
        progress_label="Regret-target scaling variants",
        plot_title_prefix="ESCHER corrected regret-target scaling ablation",
        worker_module=(
            "experiments.leduc_poker.escher_regret_target_scale_only_ablation.run"
        ),
        paired_delta_fields=(
            list(DEFAULT_PAIRED_DELTA_FIELDS)
            + PROCESSING_DIAGNOSTIC_DELTA_FIELDS
        ),
        additional_paired_baseline_ids=[CORRECTED_STANDARDIZED_CONTROL_ID],
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
