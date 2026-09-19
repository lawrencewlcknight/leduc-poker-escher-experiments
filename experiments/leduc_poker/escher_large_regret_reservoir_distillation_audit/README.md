# Experiment 47: One-Million-Row Regret-Reservoir Audit

## Purpose

Experiment 47 is a controlled derivative of Experiment 45. The capacity of
each player's regret replay reservoir increases from 50,000 to 1,000,000 rows.
The lifetime average-policy reservoir remains at 1,000,000 rows, while the
transient value-training and value-validation buffers remain at 50,000 rows.
No traversal, optimisation, network, seed, endpoint or distillation setting is
changed.

The three source seeds (`1234`, `2025`, and `31415`) each train for 12 active
hours. The same four frozen-average-policy-reservoir fitting arms are then run,
so the output remains directly comparable with Experiment 45. The primary
comparison is Experiment 47 versus Experiment 45: it tests whether loss of
older regret observations contributed to standard ESCHER's plateau.

## Local smoke test

```bash
./gcp/run_escher_large_regret_reservoir_distillation_audit.sh smoke-local
```

## GCP Batch

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp47-regret1m-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_escher_large_regret_reservoir_distillation_audit.sh run
```

The cloud controller performs smoke, launches the three seeds concurrently on
separate `n2-standard-8` VMs, and aggregates their outputs. It is independent
of the submitting laptop after submission. Monitor or resume with:

```bash
./gcp/run_escher_large_regret_reservoir_distillation_audit.sh status
./gcp/run_escher_large_regret_reservoir_distillation_audit.sh resume
```

Each training task has a 24-hour hard ceiling. The intended source-training
budget is 12 hours; the remaining allowance covers the completed boundary
iteration and frozen-reservoir distillation.

The analysis directory includes the inherited raw and summarised source-policy
trajectories and exploitability charts by both training time and nodes touched.
