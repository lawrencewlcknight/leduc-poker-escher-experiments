# Experiment 48: One-Million-Row All-Memory Audit

## Purpose

Experiment 48 extends Experiment 47 by increasing the transient value-training
and value-validation buffer capacities from 50,000 to 1,000,000 rows. The two
regret reservoirs and the lifetime average-policy reservoir are already one
million rows, so every effective in-memory reservoir or buffer has a
one-million-row ceiling.

Everything else is inherited from Experiment 45: seeds `1234`, `2025`, and
`31415`; 12 active source-training hours; the `256x256x128` policy network; and
the four frozen-reservoir distillation arms. The primary comparison is
Experiment 48 versus Experiment 47, isolating the value-buffer capacity change.

The value buffers are transient and cleared at the configured iteration
boundary. One million is therefore a ceiling, not a guarantee that the buffers
will contain one million rows. The saved configuration and source summaries
report all four effective capacities explicitly.

## Local smoke test

```bash
./gcp/run_escher_all_large_memory_distillation_audit.sh smoke-local
```

## GCP Batch

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp48-all1m-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_escher_all_large_memory_distillation_audit.sh run
```

The cloud controller performs smoke, launches the three seeds concurrently on
separate `n2-standard-8` VMs, and aggregates their outputs. It is independent
of the submitting laptop after submission. Monitor or resume with:

```bash
./gcp/run_escher_all_large_memory_distillation_audit.sh status
./gcp/run_escher_all_large_memory_distillation_audit.sh resume
```

Each training task has a 24-hour hard ceiling. The intended source-training
budget is 12 hours; the remaining allowance covers the completed boundary
iteration and frozen-reservoir distillation.

The analysis directory includes the inherited raw and summarised source-policy
trajectories and exploitability charts by both training time and nodes touched.
