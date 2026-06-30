"""Checks for recent-experiment GCP smoke-test scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recent_smoke_scripts_are_valid_bash():
    for script in [
        ROOT / "gcp" / "run_recent_experiment_smoke_tests.sh",
        ROOT / "gcp" / "submit_recent_experiment_smoke_tests.sh",
    ]:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_recent_smoke_runner_covers_experiments_23_to_27():
    script = (
        ROOT / "gcp" / "run_recent_experiment_smoke_tests.sh"
    ).read_text(encoding="utf-8")

    for module in [
        "escher_regret_target_processing_ablation",
        "escher_action_head_residual_mlp_sweep",
        "escher_average_policy_weighting_ablation",
        "escher_factorised_regret_head_ablation",
        "escher_action_head_layer_norm_residual_ablation",
    ]:
        assert module in script
