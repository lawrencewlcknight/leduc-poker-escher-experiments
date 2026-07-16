"""Configuration for Experiment 38's regret replay composition ablation."""

from __future__ import annotations

from copy import deepcopy

from escher_poker.replay import (
    ALL_SAMPLES,
    COUNTERFACTUAL_REACH_WEIGHTED,
    INFOSET_STRATIFIED,
    RARE_HISTORY_QUOTA,
    RESERVOIR,
)
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)


DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)
BASELINE_VARIANT_ID = "experiment_28_reservoir_replay"
RARE_HISTORY_QUOTA_PER_INFOSET = 64
COUNTERFACTUAL_REACH_WEIGHT_FLOOR = 1e-6


def _variant(variant_id, variant_label, variant_description, replay_mode):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "regret_replay_mode": replay_mode,
        "regret_replay_rare_history_quota": RARE_HISTORY_QUOTA_PER_INFOSET,
        "regret_replay_weight_floor": COUNTERFACTUAL_REACH_WEIGHT_FLOOR,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Experiment 28 reservoir replay",
        "Exact Experiment 28 global uniform reservoir for regret samples.",
        RESERVOIR,
    ),
    _variant(
        "all_regret_samples",
        "Store every regret sample",
        (
            "Append-only regret replay with no finite capacity; value and "
            "average-policy replay remain unchanged."
        ),
        ALL_SAMPLES,
    ),
    _variant(
        "infoset_stratified_replay",
        "Infoset-stratified replay",
        (
            "Equal-capacity per-infoset reservoirs within Experiment 28's "
            "global regret replay capacity."
        ),
        INFOSET_STRATIFIED,
    ),
    _variant(
        "rare_history_quota_replay",
        "Rare-history quota replay",
        (
            "Protects up to 64 samples per infoset and assigns remaining "
            "capacity to a global overflow reservoir."
        ),
        RARE_HISTORY_QUOTA,
    ),
    _variant(
        "counterfactual_reach_weighted_replay",
        "Counterfactual-reach-weighted replay",
        (
            "Weighted priority reservoir using opponent-and-chance reach at "
            "each sampled player infoset."
        ),
        COUNTERFACTUAL_REACH_WEIGHTED,
    ),
]


DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_replay_composition_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "regret_replay_mode": RESERVOIR,
    "regret_replay_rare_history_quota": RARE_HISTORY_QUOTA_PER_INFOSET,
    "regret_replay_weight_floor": COUNTERFACTUAL_REACH_WEIGHT_FLOOR,
})
