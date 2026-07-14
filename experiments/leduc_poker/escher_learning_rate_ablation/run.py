"""CLI for Experiment 30 ESCHER candidate learning-rate ablation."""

from __future__ import annotations

from typing import List, Optional

from experiments.leduc_poker.escher_variant_ablation_runner import run_variant_ablation

from .config import BASELINE_VARIANT_ID, DEFAULT_CONFIG, DEFAULT_SEEDS, VARIANTS


def main(argv: Optional[List[str]] = None) -> int:
    return run_variant_ablation(
        argv,
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        variants=VARIANTS,
        baseline_variant_id=BASELINE_VARIANT_ID,
        output_root="outputs/learning_rate_ablation",
        description="Run the Leduc poker ESCHER candidate learning-rate ablation.",
        logger_name="escher_poker.experiment.learning_rate_ablation",
        progress_label="Learning-rate variants",
        plot_title_prefix="ESCHER learning-rate ablation",
        worker_module="experiments.leduc_poker.escher_learning_rate_ablation.run",
        extra_summary_fields={
            "learning_rate_variant": "learning_rate",
        },
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
