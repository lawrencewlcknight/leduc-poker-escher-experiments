"""CLI entry point for the ESCHER action-head LayerNorm/residual-LN ablation."""

from __future__ import annotations

from experiments.leduc_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER action-head LayerNorm/residual-LN ablation.",
        output_root="outputs/action_head_layer_norm_residual_ablation",
        progress_label="Action-head LayerNorm/residual variants",
        final_plot_title="ESCHER action-head LayerNorm/residual-LN ablation: final exploitability",
        unknown_label="action-head LayerNorm/residual-LN",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
