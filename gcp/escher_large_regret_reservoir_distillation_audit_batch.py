#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for Experiment 47."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import escher_frozen_policy_distillation_audit_batch as _base


MODULE = "experiments.leduc_poker.escher_large_regret_reservoir_distillation_audit.cloud"
TASK_COUNT = 3


def build_job(args) -> dict:
    previous_module = _base.MODULE
    try:
        _base.MODULE = MODULE
        job = _base.build_job(args)
    finally:
        _base.MODULE = previous_module

    runnable = job["taskGroups"][0]["taskSpec"]["runnables"][0]
    script = runnable["script"]["text"]
    replacements = {
        "gcp/run_escher_frozen_policy_distillation_audit.sh": (
            "gcp/run_escher_large_regret_reservoir_distillation_audit.sh"
        ),
        "EXP45_REMOTE_CONTROLLER": "EXP47_REMOTE_CONTROLLER",
        "exp45-controller": "exp47-controller",
        "Experiment 45": "Experiment 47",
    }
    for source, target in replacements.items():
        script = script.replace(source, target)
    runnable["script"]["text"] = script
    job["labels"] = {
        "experiment": "escher-regret-reservoir-audit",
        "stage": args.kind,
    }
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("controller", "smoke", "train", "aggregate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--repo-url", default=_base.REPO_URL)
    parser.add_argument("--parallelism", type=int, default=TASK_COUNT)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--controller-action",
        choices=("orchestrate", "orchestrate-resume"),
        default="orchestrate",
    )
    args = parser.parse_args()
    if args.parallelism < 1 or args.parallelism > TASK_COUNT:
        parser.error(f"--parallelism must be between 1 and {TASK_COUNT}")
    if args.kind == "controller" and (not args.project_id or not args.region):
        parser.error("controller jobs require --project-id and --region")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(build_job(args), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
