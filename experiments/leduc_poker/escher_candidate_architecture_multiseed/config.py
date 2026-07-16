"""Configuration for the candidate ESCHER architecture multi-seed run.

Experiment 28 trains the strongest architecture suggested by the preceding
ESCHER architecture diagnostics over five fixed Leduc poker seeds. The model
combines the Experiment 27 deep plain trunk, the Experiment 22 regret action
head, and the Experiment 23 regret-target standardisation treatment.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_author_budget_multiseed.config import (
    DEFAULT_CONFIG as AUTHOR_BUDGET_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_multiseed_baseline.config import DEVELOPMENT_SEEDS_5
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config

DEFAULT_SEEDS = list(DEVELOPMENT_SEEDS_5)

CANDIDATE_VARIANT = {
    "variant_id": "deep_plain_standardized_regret_action_head",
    "variant_label": "Deep plain + standardized regret action head",
    "variant_description": (
        "Candidate ESCHER architecture: 256x256x128 plain policy, regret, "
        "and value trunks; standard linear policy output; one 64-unit "
        "per-action regret head; standardized legal regret targets."
    ),
}

BASE_CONFIG = deepcopy(AUTHOR_BUDGET_DEFAULT_CONFIG)
BASE_CONFIG.update({
    "experiment_name": "leduc_poker_escher_candidate_architecture_multiseed",
    "policy_network_layers": (256, 256, 128),
    "regret_network_layers": (256, 256, 128),
    "value_network_layers": (256, 256, 128),
    "policy_network_activation": "leakyrelu",
    "regret_network_activation": "leakyrelu",
    "value_network_activation": "leakyrelu",
    "policy_network_layer_norm": False,
    "regret_network_layer_norm": False,
    "value_network_layer_norm": False,
    "policy_network_residual_mode": "none",
    "regret_network_residual_mode": "none",
    "value_network_residual_mode": "none",
    "policy_network_head_depth": 0,
    "policy_network_head_units": None,
    "regret_network_head_depth": 1,
    "regret_network_head_units": 64,
    "regret_network_output_mode": "direct",
    "regret_target_baseline": "author_state_value",
    "regret_target_processing": "standardize",
    "regret_target_clip_value": 1.0,
    "regret_target_standardize_epsilon": 1e-6,
    "regret_replay_mode": "reservoir",
    "regret_replay_rare_history_quota": 64,
    "regret_replay_weight_floor": 1e-6,
    "use_balanced_probs": False,
    "balanced_sampling_mix": 0.0,
    "track_sampling_coverage": False,
})

DEFAULT_CONFIG = make_variant_config(BASE_CONFIG, CANDIDATE_VARIANT)
