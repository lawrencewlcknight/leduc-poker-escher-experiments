"""CLI for Experiment 32 ESCHER regret/value work-balance ablation."""

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
        output_root="outputs/regret_value_work_ablation",
        description="Run the Leduc poker ESCHER regret/value work-balance ablation.",
        logger_name="escher_poker.experiment.regret_value_work_ablation",
        progress_label="Regret/value work variants",
        plot_title_prefix="ESCHER regret/value work ablation",
        extra_summary_fields={
            "regret_traversals_variant": "num_traversals",
            "value_traversals_variant": "num_val_fn_traversals",
            "regret_train_steps_variant": "regret_network_train_steps",
            "value_train_steps_variant": "value_network_train_steps",
        },
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
