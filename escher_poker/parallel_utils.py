"""Pure helpers for deterministic ESCHER experience-worker orchestration."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy import stats


WORKER_SEED_STRIDE = 1_000_003


def worker_seed(run_seed: int, worker_index: int) -> int:
    """Return the deterministic distinct seed assigned to one worker."""
    worker_index = int(worker_index)
    if worker_index < 0:
        raise ValueError("worker_index must be non-negative.")
    return int(run_seed) + WORKER_SEED_STRIDE * (worker_index + 1)


def partition_total(total: int, parts: int) -> List[int]:
    """Split an integer budget exactly, with at most one item of imbalance."""
    total = int(total)
    parts = int(parts)
    if total < 0:
        raise ValueError("total must be non-negative.")
    if parts <= 0:
        raise ValueError("parts must be positive.")
    quotient, remainder = divmod(total, parts)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(parts)
    ]


def aggregate_replay_diagnostics(rows: Iterable[Dict]) -> Dict[str, float]:
    """Combine replay counters from independent equal-purpose worker shards."""
    rows = list(rows)
    stored = np.asarray([row.get("stored_count", 0) for row in rows], dtype=float)
    streamed = np.asarray([row.get("stream_count", 0) for row in rows], dtype=float)
    weights = np.asarray(
        [row.get("stored_weight_mean", np.nan) for row in rows],
        dtype=float,
    )
    finite_weight = np.isfinite(weights) & (stored > 0.0)
    total_stored = float(np.sum(stored))
    total_streamed = float(np.sum(streamed))
    return {
        "stored_count": int(total_stored),
        "stream_count": int(total_streamed),
        "retention_fraction": (
            total_stored / total_streamed if total_streamed else 0.0
        ),
        # Infosets can occur in multiple worker shards, so adding worker-local
        # cardinalities would overcount. The parallel comparison does not use
        # these replay-composition diagnostics as outcome measures.
        "unique_infosets": np.nan,
        "samples_per_infoset_min": np.nan,
        "samples_per_infoset_mean": np.nan,
        "samples_per_infoset_max": np.nan,
        "samples_per_infoset_cv": np.nan,
        "stored_weight_mean": (
            float(np.average(weights[finite_weight], weights=stored[finite_weight]))
            if np.any(finite_weight)
            else np.nan
        ),
    }


def equivalence_summary(
    deltas: Iterable[float],
    margin: float,
    *,
    alpha: float = 0.05,
) -> Dict[str, float | int | bool]:
    """Summarize paired equivalence using the standard 90% CI/TOST criterion."""
    values = np.asarray(list(deltas), dtype=np.float64)
    values = values[np.isfinite(values)]
    margin = float(margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("equivalence margin must be positive and finite.")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5).")
    if values.size == 0:
        return {
            "margin": margin,
            "n": 0,
            "mean_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "all_seed_deltas_within_margin": False,
            "tost_equivalent": False,
        }

    mean = float(np.mean(values))
    if values.size > 1:
        standard_error = float(stats.sem(values))
        critical = float(stats.t.ppf(1.0 - alpha, df=values.size - 1))
        half_width = critical * standard_error
    else:
        half_width = np.inf
    lower = mean - half_width
    upper = mean + half_width
    return {
        "margin": margin,
        "n": int(values.size),
        "mean_delta": mean,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "all_seed_deltas_within_margin": bool(np.all(np.abs(values) <= margin)),
        "tost_equivalent": bool(lower > -margin and upper < margin),
    }
