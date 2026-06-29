"""CLI entry point for the ESCHER factorised regret-head ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER factorised regret-head ablation.",
        output_root="outputs/factorised_regret_head_ablation",
        progress_label="Factorised regret-head variants",
        final_plot_title="ESCHER factorised regret-head ablation: final exploitability",
        unknown_label="factorised regret-head",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
