"""CLI for Experiment 33 ESCHER regret action-head capacity ablation."""

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
        output_root="outputs/regret_action_head_capacity_ablation",
        description="Run the Leduc poker ESCHER regret action-head capacity ablation.",
        logger_name="escher_poker.experiment.regret_action_head_capacity_ablation",
        progress_label="Regret action-head variants",
        plot_title_prefix="ESCHER regret action-head capacity ablation",
        worker_module="experiments.leduc_poker.escher_regret_action_head_capacity_ablation.run",
        extra_summary_fields={
            "regret_head_depth_variant": "regret_network_head_depth",
            "regret_head_units_variant": "regret_network_head_units",
        },
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
