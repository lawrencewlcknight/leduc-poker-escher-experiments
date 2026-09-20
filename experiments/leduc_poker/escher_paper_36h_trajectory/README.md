# Experiment 49: 36-Hour Paper-Aligned ESCHER Trajectory

## Purpose

Experiment 49 confirms the best standard-ESCHER configuration identified by
Experiment 46 over a longer horizon. It retains the paper-aligned traversal,
minibatch and network-training settings, the one-million-row average-policy
reservoir, and the selected grouped soft-target cross-entropy policy fit.

Five seeds (`1234`, `2025`, `31415`, `27182`, and `16180`) each receive exactly
36 active learner hours. Each seed runs on a separate `n2-standard-8` VM. The
experiment freezes the average-policy reservoir at the first completed
iteration crossing every 30-minute active-time boundary, the first completed
iteration crossing 15 million nodes, and the final completed iteration at the
36-hour boundary.

Checkpoint persistence is excluded from the learner clock. No average-policy
network is fitted during timed training. After every training worker succeeds,
a separate five-task evaluation stage fits the selected grouped cross-entropy
policy from a common seed-specific initialization at every frozen checkpoint
and evaluates exact Leduc exploitability. This separation ensures evaluation
cannot reduce the learner's 36-hour budget and permits failed evaluation jobs
to resume without repeating training.

## Scientific contract

- Five thesis-comparison seeds, one VM per seed.
- 36 active learner hours per seed; an in-flight iteration is completed.
- Frozen policy-reservoir checkpoints every 30 active minutes.
- Explicit first-completed-iteration endpoint crossing 15 million nodes.
- Explicit final-completed-iteration endpoint at 36 hours.
- Experiment 46 paper-aligned learner hyperparameters.
- One-million-row average-policy reservoir.
- Grouped, iteration-weighted soft-target cross-entropy policy fitting using
  the Experiment 46 matched-example budget.
- Exact neural and empirical-reservoir exploitability at every checkpoint.
- No automatic Batch retry for the expensive training/evaluation arrays; use
  the resumable controller so completed seed artifacts are not repeated.

## Mandatory smoke test

From the repository root and activated project environment:

```bash
./gcp/run_escher_paper_36h_trajectory.sh smoke-local
```

The smoke test exercises training, frozen-reservoir checksums, deferred fitting,
exact evaluation, endpoint selection, aggregation, tables, and all three charts.

## GCP run

Reuse the existing `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` values:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp49-paper36h-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_escher_paper_36h_trajectory.sh run
```

The cloud-owned controller runs a cloud smoke, five parallel training tasks,
five parallel deferred-evaluation tasks, and aggregation. The laptop may be
disconnected after the controller is submitted. Monitor or resume with:

```bash
./gcp/run_escher_paper_36h_trajectory.sh status
./gcp/run_escher_paper_36h_trajectory.sh resume
```

Training workers have a 48-hour hard ceiling and evaluation workers a 20-hour
hard ceiling. These are cost guards, not expected runtimes. Training is expected
to finish a little above 36 elapsed hours; deferred fitting is expected to add
roughly 10 hours, with seeds running concurrently in both stages.

## Outputs

The small, thesis-ready artifacts are under `$BUCKET/$RUN_ID/analysis/`:

- `checkpoint_metrics.csv`: every seed/checkpoint neural and empirical result;
- `trajectory_summary.csv`: time-aligned cross-seed statistics;
- `trajectory_by_nodes_summary.csv`: node-aligned interpolated statistics;
- `exploitability_by_training_time.png`;
- `exploitability_by_nodes.png`;
- `distillation_gap_by_training_time.png`;
- `endpoint_15m_seed_metrics.csv` and `endpoint_36h_seed_metrics.csv`;
- `endpoint_summary.csv`;
- `thesis_15m_seed_summary.csv` and `thesis_15m_aggregate_summary.csv`;
- `aggregate_summary.json`.

The `training/` cloud prefix retains the frozen reservoirs. The `evaluation/`
prefix retains resumable per-checkpoint metric files and playable weights for
the 15-million-node and 36-hour endpoint policies. Download only `analysis/`
for ordinary interpretation; retain the larger worker artifacts in Cloud
Storage for reproducibility and recovery.
