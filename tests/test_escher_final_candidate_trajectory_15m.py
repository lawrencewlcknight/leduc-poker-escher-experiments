"""Configuration and export checks for Experiment 44."""

import csv

import numpy as np

from escher_poker.experiment_utils import (
    export_trajectory_history,
    normalised_auc_over_range,
)
from experiments.leduc_poker.escher_final_candidate_trajectory_15m.config import (
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPECTED_TRAJECTORY_POINTS,
    EXPERIMENT_43_CONFIG,
    TARGET_NUM_ITERATIONS,
    TARGET_SOLVE_PASSES,
    expected_trajectory_points,
    validate_config,
)


def test_trajectory_experiment_preserves_selected_algorithm_and_budget():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert TARGET_NUM_ITERATIONS == 1_300
    assert TARGET_SOLVE_PASSES == 1_301
    assert DEFAULT_CONFIG["check_exploitability_every"] == 10
    assert EXPECTED_TRAJECTORY_POINTS == 132
    assert DEFAULT_CONFIG["record_initial_policy_evaluation"] is True
    assert DEFAULT_CONFIG["trajectory_auc_start_nodes"] == 0
    assert DEFAULT_CONFIG["trajectory_auc_end_nodes"] == 15_000_000
    assert DEFAULT_CONFIG["final_window_width_nodes"] == 1_000_000

    permitted_changes = {
        "experiment_name",
        "trajectory_history_filename",
        "trajectory_summary_filename",
        "trajectory_alignment_key",
        "trajectory_x_axis",
        "record_initial_policy_evaluation",
        "trajectory_auc_start_nodes",
        "trajectory_auc_end_nodes",
        "trajectory_auc_hold_boundaries",
        "final_window_width_nodes",
    }
    for key, value in EXPERIMENT_43_CONFIG.items():
        if key not in permitted_changes:
            assert DEFAULT_CONFIG[key] == value


def test_expected_trajectory_points_includes_first_and_final_evaluations():
    config = dict(DEFAULT_CONFIG, num_iterations=20, check_exploitability_every=5)
    assert expected_trajectory_points(config) == 6
    validate_config(config)


def _fake_result(seed, exploitability):
    points = len(exploitability)
    values = np.arange(points, dtype=float)
    diagnostics = {
        "policy_loss": values,
        "value_loss": values,
        "value_test_loss": values,
        "regret_loss_player_0": values,
        "regret_loss_player_1": values,
        "average_policy_buffer_size": values.astype(int),
        "regret_buffer_size_player_0": values.astype(int),
        "regret_buffer_size_player_1": values.astype(int),
        "value_buffer_size": values.astype(int),
        "value_test_buffer_size": values.astype(int),
    }
    return {
        "seed": seed,
        "iterations": np.arange(points, dtype=int) * 10,
        "nodes_touched": np.arange(1, points + 1, dtype=float) * 100,
        "wall_clock_seconds": values,
        "exploitability": np.asarray(exploitability, dtype=float),
        "average_policy_value": values,
        "policy_value_error": values,
        "diagnostics": diagnostics,
    }


def test_trajectory_export_writes_raw_and_cross_seed_summary(tmp_path):
    results = [_fake_result(1, [0.5, 0.3]), _fake_result(2, [0.7, 0.1])]
    export_trajectory_history(tmp_path, results)

    with open(tmp_path / "trajectory_history.csv", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    with open(tmp_path / "trajectory_summary.csv", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    assert len(raw_rows) == 4
    assert raw_rows[0]["checkpoint_index"] == "0"
    assert raw_rows[0]["is_initial_policy_evaluation"] == "False"
    assert len(summary_rows) == 2
    assert float(summary_rows[0]["mean_exploitability"]) == 0.6
    assert float(summary_rows[0]["mean_wall_clock_seconds"]) == 0.0
    assert int(summary_rows[0]["n_seeds"]) == 2


def test_normalised_auc_uses_explicit_zero_to_15m_range_and_boundaries():
    nodes = [0, 5_000_000, 10_000_000, 14_900_000]
    exploitability = [1.0, 2 / 3, 1 / 3, 0.0]
    value = normalised_auc_over_range(
        nodes,
        exploitability,
        0,
        15_000_000,
        hold_boundary_values=True,
    )
    assert np.isfinite(value)
    assert 0.49 < value < 0.51
