"""CLI for Experiment 31 ESCHER replay-capacity ablation."""

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
        output_root="outputs/replay_capacity_ablation",
        description="Run the Leduc poker ESCHER replay-capacity ablation.",
        logger_name="escher_poker.experiment.replay_capacity_ablation",
        progress_label="Replay-capacity variants",
        plot_title_prefix="ESCHER replay-capacity ablation",
        worker_module="experiments.leduc_poker.escher_replay_capacity_ablation.run",
        extra_summary_fields={
            "replay_capacity_variant": "memory_capacity",
            "regret_batch_variant": "batch_size_regret",
            "value_batch_variant": "batch_size_value",
        },
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
