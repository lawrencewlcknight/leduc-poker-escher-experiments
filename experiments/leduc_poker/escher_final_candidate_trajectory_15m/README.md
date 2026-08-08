# Experiment 44: ESCHER 15M-Node Trajectory Export

## Purpose

This experiment reruns the selected ESCHER configuration used by Experiment
43 and saves the complete in-training evaluation trajectory. The algorithm,
architecture, five seeds, and training budget are unchanged. The difference is
output persistence: the exact untrained policy at node zero and all 131
in-training evaluations per seed are exported instead of only the five
temporal policy snapshots.

The first saved point is an exact evaluation of the untrained policy at zero
nodes and zero training time. In-training evaluations occur every 10 configured
iterations through iteration 1,300, giving approximately 15 million nodes per
seed at the endpoint and 132 total saved points per seed.

The seed summary reports two explicitly budget-aligned metrics:

- `final_window_mean_exploitability` is the node-weighted mean over 14M--15M
  nodes, calculated by trapezoidal integration.
- `normalised_auc_exploitability_0_to_target_nodes` is exploitability AUC over
  0--15M nodes divided by 15M. Boundaries inside the trajectory are linearly
  interpolated. If stochastic node counts finish slightly below 15M, the last
  observation is carried forward over the small remaining interval; the
  observed coverage fraction is also recorded.

## Full run

From the repository root:

```bash
python -m experiments.leduc_poker.escher_final_candidate_trajectory_15m.run
```

The default output root is `outputs/final_candidate_trajectory_15m/`. The
runner rewrites the trajectory and seed-summary files after each seed so that
completed seeds remain usable if a later seed fails.

For the same GCP Batch setup used by Experiment 43:

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-escher-exp44-trajectory-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_final_candidate_trajectory_15m.run \
    --output-root outputs/cloud/leduc-escher-exp44-trajectory" \
  "n2-standard-8" "345600" "8000" "32000" "100"
```

## Local smoke test

```bash
python -m experiments.leduc_poker.escher_final_candidate_trajectory_15m.run \
  --seeds 1234 \
  --iterations 2 \
  --evaluation-interval 1 \
  --traversals 2 \
  --value-traversals 2 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --policy-network-layers 8,8 \
  --regret-network-layers 8,8 \
  --value-network-layers 8,8 \
  --batch-size-regret 2 \
  --batch-size-value 2 \
  --batch-size-average-policy 2 \
  --memory-capacity 128 \
  --output-root outputs/smoke_tests
```

## Principal outputs

| File | Contents |
| --- | --- |
| `trajectory_history.csv` | One row per seed and evaluation, including iteration, actual nodes, exploitability, policy value, elapsed time, losses, and replay sizes. |
| `trajectory_summary.csv` | Cross-seed node, wall-clock and exploitability summaries at every evaluation index. |
| `seed_summary.csv` | Endpoint, 14M--15M final-window mean, normalized 0--15M AUC, runtime and best metrics for every completed seed. |
| `aggregate_summary.json` | Cross-seed statistics for the seed summaries. |
| `experiment_metadata.json` | Exact configuration and seed list. |
| `exploitability_by_nodes_multiseed.png` | Mean exploitability with standard-error band and faint seed trajectories; x-axis begins at zero. |
| `exploitability_by_time_multiseed.png` | Exploitability against training wall-clock hours, including individual seeds and mean standard-error band. |

For the four-algorithm comparison, use `trajectory_history.csv` as the raw
ESCHER input or `trajectory_summary.csv` if only the mean and uncertainty are
required.
