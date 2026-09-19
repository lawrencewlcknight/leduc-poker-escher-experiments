# Experiment 45: Frozen-Reservoir Distillation Audit

## Purpose

This experiment tests whether standard ESCHER's apparent Leduc plateau is
substantially caused by average-policy approximation rather than by the
underlying regret learner.

Three development seeds (`1234`, `2025`, and `31415`) use the selected standard
ESCHER architecture and each receives exactly 12 active training hours on an
`n2-standard-8`. Touched nodes and completed iterations are outcomes rather
than stopping criteria. The solver checks the monotonic deadline between
iterations and completes an iteration already in flight, so a run can exceed
12 hours by at most one iteration. Intermediate average-policy fitting and
exact exploitability evaluation are disabled: the timed budget is reserved
for regret and history-value learning. Learner-only telemetry (time, nodes,
losses and replay occupancy) is recorded every ten completed iterations at
negligible cost. The playable source policy is fitted once after the timed
boundary. The regret and transient value memories remain at 50,000 rows. Only
the lifetime average-policy reservoir is enlarged to 1,000,000 rows.

After each source run, the final reservoir is frozen to a compressed NPZ file,
checksummed, reloaded, grouped by information set, and evaluated exactly. Four
fresh `256x256x128` policy networks are then fitted from the identical frozen
source and identical initialization:

1. row-wise iteration-weighted MSE at the existing fitting budget;
2. row-wise iteration-weighted soft-target cross-entropy with the same number
   of network-example presentations;
3. grouped iteration-weighted soft-target cross-entropy with matched
   network-example presentations;
4. the grouped objective with four times as many presentations.

For grouped fitting, one row is the sufficient statistic for an observed
information set. Its objective weight makes grouped soft-target
cross-entropy algebraically identical to row-wise soft-target
cross-entropy. The matched-example arms control the number of actual network
input rows processed, not merely the number of optimiser steps.

## Local smoke test

From the repository root:

```bash
python -m experiments.leduc_poker.escher_frozen_policy_distillation_audit.run \
  --smoke \
  --output-root outputs/smoke_tests
```

The smoke test runs one tiny seed, freezes and reloads its reservoir, fits all
four arms, performs exact Leduc evaluations, and writes both charts.

## Full local run

```bash
python -m experiments.leduc_poker.escher_frozen_policy_distillation_audit.run
```

The three seeds run sequentially locally. Expect approximately 39--48 elapsed
hours on hardware comparable to `n2-standard-8`: 36 active training hours plus
final policy fitting and the frozen-reservoir arms. Results are rewritten after
every completed seed so a later failure does not erase earlier evidence.

## GCP Batch

Experiment 45 uses the same cloud-owned controller pattern as Experiment 35
in the ESCHER-architecture repository. The controller first requires the cloud
smoke test to succeed, then submits the three-seed training array, and finally
runs aggregation. The array has `taskCount=3`, `parallelism=3`, and
`taskCountPerNode=1`, so the three isolated seed tasks execute concurrently on
three separate `n2-standard-8` VMs. Aggregation cannot start until all three
have succeeded.

Run the mandatory local smoke test from the repository root:

```bash
./gcp/run_escher_frozen_policy_distillation_audit.sh smoke-local
```

After pushing that exact tested commit, reuse the `PROJECT_ID`, `REGION`,
`BUCKET`, and `SA_EMAIL` environment variables from earlier experiments:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp45-dist-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_escher_frozen_policy_distillation_audit.sh run
```

The remote controller must be able to create and inspect its child Batch jobs
and act as the configured runner account. The launcher now checks these roles
before submission and prints the exact one-time IAM commands if either is
missing. The required bindings are `roles/batch.jobsEditor` on the project and
`roles/iam.serviceAccountUser` on `SA_EMAIL` for that same service account.

The laptop may be disconnected after submission. Check or resume the
cloud-owned workflow with:

```bash
./gcp/run_escher_frozen_policy_distillation_audit.sh status
./gcp/run_escher_frozen_policy_distillation_audit.sh resume
```

Each seed task has a 24-hour hard ceiling and one automatic retry; the full
controller has a seven-day ceiling. Expected elapsed time is approximately
14--17 hours, comprising setup, the slowest source-training and distillation
worker, and final aggregation. The compute budget remains approximately
39--48 N2 VM-hours because the three workers consume those hours concurrently.
A completed seed is uploaded independently and is reused by `resume`, so a
later-stage failure does not require successful earlier seeds to be rerun.

## Principal outputs

| Output | Contents |
| --- | --- |
| `seed_<seed>/frozen_average_policy_reservoir.npz` | Lossless frozen final reservoir used by every arm. |
| `seed_<seed>/source_training_progress.csv` | Learner-only time, nodes, losses and replay sizes; it does not fit or evaluate the average-policy network. |
| `source_seed_metrics.csv` | Source training budget, realised time/overshoot, iterations, nodes, reservoir, empirical-policy and archived neural-policy diagnostics. |
| `source_training_progress.csv` | Combined learner-only progress records for all seeds. |
| `source_nodes_by_training_time.png` | Per-seed source-learning throughput over the timed budget. |
| `source_losses_by_training_time.png` | Per-seed regret and history-value losses without policy-distillation overhead. |
| `fit_metrics.csv` | Exact exploitability, empirical gap, fitting work and runtime for every arm and seed. |
| `arm_summary.csv` | Cross-seed means, standard deviations and standard errors. |
| `exploitability_by_distillation_arm.png` | Exact arm comparison with the empirical-reservoir reference. |
| `distillation_gap_by_arm.png` | Neural minus empirical-reservoir exploitability. |
| `aggregate_summary.json` | Machine-readable experiment summary. |

The source policy and four offline distillation arms have endpoint
exploitability results only. Experiment 44 remains the dedicated dense
source-policy trajectory study; this audit deliberately prioritises a clean
12-hour learner budget.
