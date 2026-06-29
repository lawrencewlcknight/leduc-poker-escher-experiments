"""Configuration for the ESCHER regret-target processing ablation.

The experiment uses the carried-forward ESCHER configuration: the Experiment
13 training protocol with ``(256, 128)`` policy, regret, and value trunks, no
importance-sampling correction, uniform zero-regret fallback, and one 64-unit
separate per-action head on the regret networks. Variants differ only in how
sampled legal regret targets are processed before the supervised regret-network
loss. Replay memories retain raw sampled regret targets for all variants, and
illegal-action targets remain masked to zero during processing.
"""

from __future__ import annotations

from experiments.leduc_poker.escher_architecture_base import make_default_config

DEFAULT_SEEDS = [1234, 2025, 31415]
DEFAULT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]


def _variant(
    variant_id,
    variant_label,
    variant_description,
    regret_target_processing,
    regret_target_clip_value=1.0,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "regret_target_processing": regret_target_processing,
        "regret_target_clip_value": regret_target_clip_value,
    }


VARIANTS = [
    _variant(
        "raw_regret_targets",
        "Raw regret targets",
        "Carry-forward ESCHER baseline: raw sampled regret targets.",
        "none",
    ),
    _variant(
        "standardized_regret_targets",
        "Standardized regret targets",
        "Batch-standardizes sampled legal regret targets before the regret-network loss.",
        "standardize",
    ),
    _variant(
        "clipped_regret_targets",
        "Clipped regret targets",
        "Clips sampled legal regret targets to [-1, 1] before the regret-network loss.",
        "clip",
    ),
    _variant(
        "standardized_clipped_regret_targets",
        "Standardized + clipped regret targets",
        "Batch-standardizes sampled legal regret targets and then clips them to [-1, 1].",
        "standardize_clip",
    ),
]

BASELINE_VARIANT_ID = "raw_regret_targets"

DEFAULT_CONFIG = make_default_config(
    "leduc_poker_escher_regret_target_processing_ablation"
)
DEFAULT_CONFIG.update({
    "policy_network_head_depth": 0,
    "policy_network_head_units": None,
    "regret_network_head_depth": 1,
    "regret_network_head_units": 64,
    "regret_target_processing": "none",
    "regret_target_clip_value": 1.0,
    "regret_target_standardize_epsilon": 1e-6,
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
})
