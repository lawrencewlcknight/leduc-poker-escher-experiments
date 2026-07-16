"""Configuration for Experiment 39's fixed sampling-policy ablation."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)


DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)
BASELINE_VARIANT_ID = "experiment_28_uniform_fixed_sampling"
TEMPERED_BALANCED_MIX = 0.5


def _variant(
    variant_id,
    variant_label,
    variant_description,
    *,
    use_balanced_probs,
    balanced_sampling_mix,
):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "use_balanced_probs": bool(use_balanced_probs),
        "balanced_sampling_mix": float(balanced_sampling_mix),
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Experiment 28 uniform fixed sampling",
        "Exact Experiment 28 uniform-action fixed sampling policy.",
        use_balanced_probs=False,
        balanced_sampling_mix=0.0,
    ),
    _variant(
        "exact_balanced_fixed_sampling",
        "Exact balanced fixed sampling",
        (
            "Fixed subtree-balanced policy whose action probabilities are "
            "proportional to descendant terminal-leaf counts."
        ),
        use_balanced_probs=True,
        balanced_sampling_mix=1.0,
    ),
    _variant(
        "tempered_balanced_fixed_sampling",
        "Tempered balanced fixed sampling",
        (
            "Fixed 50/50 convex mixture of uniform actions and the exact "
            "leaf-balanced policy."
        ),
        use_balanced_probs=True,
        balanced_sampling_mix=TEMPERED_BALANCED_MIX,
    ),
]


DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_fixed_sampling_coverage_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "use_balanced_probs": False,
    "balanced_sampling_mix": 0.0,
    "track_sampling_coverage": True,
})
