"""Tests for ESCHER parallel orchestration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from escher_poker.parallel_utils import (
    aggregate_replay_diagnostics,
    equivalence_summary,
    partition_total,
    worker_seed,
)


def test_partition_total_preserves_budget_and_balances_workers():
    result = partition_total(500, 3)
    assert result == [167, 167, 166]
    assert sum(result) == 500
    assert max(result) - min(result) == 1


def test_partition_total_supports_more_workers_than_items():
    assert partition_total(2, 4) == [1, 1, 0, 0]


def test_worker_seeds_are_deterministic_and_distinct():
    assert [worker_seed(1234, index) for index in range(3)] == [
        1_001_237,
        2_001_240,
        3_001_243,
    ]


def test_aggregate_replay_diagnostics_preserves_global_capacity_and_stream():
    result = aggregate_replay_diagnostics([
        {"stored_count": 17, "stream_count": 100, "stored_weight_mean": 0.2},
        {"stored_count": 16, "stream_count": 100, "stored_weight_mean": 0.5},
    ])
    assert result["stored_count"] == 33
    assert result["stream_count"] == 200
    assert result["retention_fraction"] == pytest.approx(33 / 200)
    assert result["stored_weight_mean"] == pytest.approx((17 * 0.2 + 16 * 0.5) / 33)
    assert np.isnan(result["unique_infosets"])


def test_equivalence_summary_uses_90_percent_ci_for_tost():
    result = equivalence_summary([0.001, -0.001, 0.0], margin=0.01)
    assert result["all_seed_deltas_within_margin"] is True
    assert result["tost_equivalent"] is True


def test_equivalence_summary_rejects_wide_or_shifted_result():
    result = equivalence_summary([0.03, 0.04, 0.05], margin=0.02)
    assert result["all_seed_deltas_within_margin"] is False
    assert result["tost_equivalent"] is False
