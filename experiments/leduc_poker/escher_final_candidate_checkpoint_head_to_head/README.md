# Experiment 43: ESCHER Temporal Checkpoint Head-to-Head

## Research question

Does the reduction in exploitability observed over long-horizon training of the
best validated ESCHER configuration correspond to progressively stronger
direct-play performance?

## Design

Each of five fixed seeds trains one uninterrupted instance of the Experiment 28
configuration. The only algorithm-budget change is the training horizon:

```text
configured iterations: 1,300
actual solve passes:    1,301
checkpoints:              260, 520, 780, 1,040, 1,300
expected nodes:          ~3M, ~6M, ~9M, ~12M, ~15M
seeds:                   1234, 2025, 31415, 27182, 16180
```

The extra solve pass is intentional and matches the established ESCHER solver
convention, which executes `range(num_iterations + 1)`. Checkpoint names use
the configured loop index; every output also records completed solve passes,
the solver's internal iteration, and actual nodes touched. Actual node counts,
not the projections above, are authoritative and label all temporal charts.
The 15-million-node projection scales the completed long-horizon baseline's
mean of 9,231,941 nodes at 801 passes by `1,301 / 801`, yielding 14,994,701.

The Experiment 28 architecture, target definition and processing, replay,
fixed sampling policy, traversal work, optimizer work, learning rate, replay
capacity, and policy-evaluation cadence are inherited unchanged. A callback
saves the already fitted average-policy network at each requested checkpoint.
Training is not stopped, resumed, or restarted, and replay buffers are not
serialized, so the checkpoint mechanism does not change the training path.

## Exact evaluation and inference

Leduc is small enough to evaluate policies exactly. Every pair of checkpoints
within each seed is evaluated in both seat assignments. If `A` is the later
policy and `B` the earlier policy, the reported effect is:

```text
0.5 * (value of A as player 0 against B + value of A as player 1 against B)
```

There is no Monte Carlo game-count choice or match-sampling noise. The
independent training seed, rather than the ten correlated checkpoint pairs
within a seed, is the primary inferential unit. The primary estimand is the
mean later-versus-earlier exact EV within each seed, aggregated over five
seeds. Adjacent-checkpoint and final-versus-first contrasts are also reported.
The analysis produces 95% t intervals and exact one-sided sign-flip tests;
the ten checkpoint-pair tests are secondary and use Holm family-wise error
correction.

With five seeds the smallest possible one-sided sign-flip p-value is
`1 / 32 = 0.03125`. Effect sizes, cross-seed consistency, and confidence
intervals should therefore be interpreted alongside significance tests.

## Run

From the repository root:

```bash
# Full five-seed training and analysis
python -m experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.run

# Re-run exact analysis against an existing run
python -m experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.run analyse \
  --run-dir outputs/final_candidate_checkpoint_head_to_head/<run-directory>
```

## Local smoke test

The smoke test exercises uninterrupted snapshot capture, exact two-seat
evaluation, inferential tables, and chart generation at a deliberately tiny
budget:

```bash
python -m experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.run \
  --seeds 1234 \
  --iterations 5 \
  --checkpoint-schedule 1,2,3,4,5 \
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

A successful smoke test returns exit code zero and writes five policy snapshots,
five rows to `training_stage_metrics.csv`, 25 ordered exact head-to-head rows,
ten later-versus-earlier inference rows, and the plots listed below.

## GCP Batch

Set `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` as described in
[`docs/GCP_BATCH_EXPERIMENTS.md`](../../../docs/GCP_BATCH_EXPERIMENTS.md).

```bash
# GCP smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp43-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.run \
    --seeds 1234 \
    --iterations 5 \
    --checkpoint-schedule 1,2,3,4,5 \
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
    --output-root outputs/cloud/escher-smoke-exp43" \
  "n2-standard-4" "3600" "4000" "16000"

# Full five-seed run, approximately 15M nodes per seed
./gcp/submit_batch_experiment.sh \
  "leduc-escher-exp43-checkpoint-h2h-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_final_candidate_checkpoint_head_to_head.run \
    --output-root outputs/cloud/leduc-escher-exp43-checkpoint-h2h" \
  "n2-standard-8" "345600" "8000" "32000" "100"
```

The full experiment performs 6,505 solve passes across five seeds. The
completed long-horizon baseline averaged 16,968 seconds per 801-pass seed;
linear scaling predicts about 27,550 seconds (7.65 hours) per new seed, or
roughly 38 hours sequentially for all five. A four-day Batch timeout leaves
headroom for machine and analysis variance, and the estimate should be checked
against the first completed seed.

## Principal outputs

| File | Contents |
| --- | --- |
| `training_stage_metrics.csv` | Actual nodes, solve passes, elapsed time, and replay sizes at every snapshot. |
| `checkpoint_exploitability_metrics.csv` | Exact NashConv/2, self-play value, and value error by seed and checkpoint. |
| `head_to_head_pairwise.csv` | Exact two-seat EV for every ordered checkpoint pair. |
| `head_to_head_primary_effect_by_seed.csv` | One independent later-versus-earlier summary effect per seed. |
| `head_to_head_inference_summary.csv` | Primary, adjacent, and endpoint estimates with intervals and exact p-values. |
| `head_to_head_pairwise_inference.csv` | Secondary pair-specific estimates with Holm-adjusted p-values. |
| `aggregate_summary.json` | Machine-readable estimands, actual endpoint nodes, and inference protocol. |
| `head_to_head_later_vs_earlier.png` | Lower-triangular exact-EV matrix labelled by mean nodes touched. |
| `head_to_head_strength_vs_earlier_by_nodes.png` | Mean EV against all earlier checkpoints over nodes. |
| `head_to_head_strength_vs_previous_by_nodes.png` | Adjacent-checkpoint EV over nodes. |
| `exploitability_by_nodes.png` | Exact exploitability at the five snapshots. |
| `average_policy_value_by_nodes.png` | Average-policy value at the five snapshots. |
| `head_to_head_primary_effect_by_seed.png` | Primary seed effects and their cross-seed mean. |
| `snapshots/*.pkl` | Lightweight average-policy snapshots used for exact analysis. |
