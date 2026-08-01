"""Design and inference checks for Experiment 43."""

import numpy as np

from experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.config import (
    CHECKPOINT_SCHEDULE,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPECTED_FINAL_NODES_TOUCHED,
    EXPERIMENT_28_CONFIG,
    TARGET_NUM_ITERATIONS,
    TARGET_SOLVE_PASSES,
    validate_config,
)
from experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.statistics import (
    build_inference_tables,
    exact_one_sided_sign_flip_p,
)


def test_temporal_experiment_uses_five_seeds_and_five_even_checkpoints():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert TARGET_NUM_ITERATIONS == 1_300
    assert TARGET_SOLVE_PASSES == 1_301
    assert CHECKPOINT_SCHEDULE == (260, 520, 780, 1_040, 1_300)
    assert DEFAULT_CONFIG["checkpoint_schedule"] == CHECKPOINT_SCHEDULE
    assert EXPECTED_FINAL_NODES_TOUCHED == 15_000_000
    assert DEFAULT_CONFIG["temporal_x_axis"] == "nodes_touched"
    assert DEFAULT_CONFIG["run_monte_carlo_validation"] is False


def test_temporal_experiment_inherits_experiment_28_algorithm_exactly():
    permitted_changes = {
        "experiment_name",
        "num_iterations",
        "checkpoint_schedule",
        "target_solve_pass_count",
        "expected_final_nodes_touched",
        "equivalence_epsilon",
        "temporal_x_axis",
        "require_complete_checkpoint_schedule",
        "annotate_heatmap",
        "head_to_head_evaluation",
        "run_monte_carlo_validation",
        "num_mc_episodes",
        "intermediate_policy_training_events_expected",
        "final_policy_training_events_expected",
        "total_policy_training_events_expected",
        "policy_gradient_steps_expected",
    }
    for key, value in EXPERIMENT_28_CONFIG.items():
        if key not in permitted_changes:
            assert DEFAULT_CONFIG[key] == value


def test_checkpoint_schedule_must_align_with_policy_evaluation():
    bad_config = dict(DEFAULT_CONFIG)
    bad_config["checkpoint_schedule"] = (261, 520, 780, 1_040, 1_300)
    try:
        validate_config(bad_config)
    except ValueError as exc:
        assert "incompatible checkpoints" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected incompatible checkpoint schedule to fail")


def _pairwise_rows(schedule, seeds):
    rows = []
    for seed in seeds:
        for checkpoint_a in schedule:
            for checkpoint_b in schedule:
                rows.append({
                    "seed": seed,
                    "checkpoint_a": checkpoint_a,
                    "checkpoint_b": checkpoint_b,
                    "A_EV_seat_averaged": 0.001 * (checkpoint_a - checkpoint_b),
                })
    return rows


def test_inference_uses_seed_as_unit_and_holm_corrects_ten_pairs():
    seeds = [1, 2, 3, 4, 5]
    seed_rows, summary_rows, pair_rows = build_inference_tables(
        _pairwise_rows(CHECKPOINT_SCHEDULE, seeds),
        CHECKPOINT_SCHEDULE,
    )
    assert len(seed_rows) == 5
    assert all(row["num_later_vs_earlier_pairs"] == 10 for row in seed_rows)
    assert len(summary_rows) == 3
    assert all(row["n_seeds"] == 5 for row in summary_rows)
    assert len(pair_rows) == 10
    assert all(np.isfinite(row["holm_adjusted_p"]) for row in pair_rows)


def test_five_consistently_positive_seed_effects_have_minimum_sign_flip_p():
    assert exact_one_sided_sign_flip_p([1, 2, 3, 4, 5]) == 1 / 32
