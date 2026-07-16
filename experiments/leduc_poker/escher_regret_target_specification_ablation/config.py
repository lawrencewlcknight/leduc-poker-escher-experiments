"""Configuration for the Experiment 28 ESCHER regret-target ablation.

The baseline exactly preserves Experiment 28's author-code target. The
treatment changes only the raw instantaneous regret baseline to Equation 7 /
Algorithm 2 of the ESCHER paper.
"""

from __future__ import annotations

from copy import deepcopy

from escher_poker.regret_targets import (
    AUTHOR_STATE_VALUE,
    PAPER_POLICY_WEIGHTED_Q,
)
from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as CANDIDATE_DEFAULT_CONFIG,
    DEFAULT_SEEDS as CANDIDATE_DEFAULT_SEEDS,
)


DEFAULT_SEEDS = list(CANDIDATE_DEFAULT_SEEDS)

BASELINE_VARIANT_ID = "author_state_value_baseline"


def _variant(variant_id, variant_label, variant_description, regret_target_baseline):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "regret_target_baseline": regret_target_baseline,
    }


VARIANTS = [
    _variant(
        BASELINE_VARIANT_ID,
        "Author-code state-value target",
        (
            "Exact Experiment 28 candidate using the public authors' "
            "Q_hat(h,a) - V_hat(h) regret target."
        ),
        AUTHOR_STATE_VALUE,
    ),
    _variant(
        "paper_policy_weighted_q",
        "Paper policy-weighted-Q target",
        (
            "Experiment 28 candidate with Equation 7 / Algorithm 2 target "
            "Q_hat(h,a) - sum_a pi(a|h) Q_hat(h,a)."
        ),
        PAPER_POLICY_WEIGHTED_Q,
    ),
]


DEFAULT_CONFIG = deepcopy(CANDIDATE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_escher_regret_target_specification_ablation",
    "variant_id": BASELINE_VARIANT_ID,
    "variant_label": VARIANTS[0]["variant_label"],
    "variant_description": VARIANTS[0]["variant_description"],
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "ablation_variants": tuple(VARIANTS),
    "regret_target_baseline": AUTHOR_STATE_VALUE,
})
