"""CLI entry point for residual trunks in the ESCHER action-head model."""

from __future__ import annotations

from experiments.leduc_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER action-head residual-MLP sweep.",
        output_root="outputs/action_head_residual_mlp_sweeps",
        progress_label="Action-head residual variants",
        final_plot_title="ESCHER action-head residual-MLP sweep: final exploitability",
        unknown_label="action-head residual-MLP",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
