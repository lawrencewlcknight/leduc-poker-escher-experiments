# Experiment 42: 20x-node long-horizon candidate ablation

This experiment tests whether substantially longer training changes the
relative performance of Experiment 28 and the evidence-weighted candidate.
Both arms use seeds `1234`, `2025`, and `31415`:

| Arm | Regret target | Target processing | Regret replay | Fixed sampling |
|---|---|---|---|---|
| Experiment 28 long-run baseline | Author state value | Batch-centred standardization | Global reservoir | Uniform actions |
| Long-run candidate | Policy-weighted Q | Batch-centred standardization | Infoset-stratified | Uniform actions |

Everything except the candidate's two requested algorithm changes and the
training horizon is inherited exactly from Experiment 28. In particular,
traversals per pass, optimizer work per pass, architecture, replay capacity,
learning rate, evaluation interval, network reinitialization, and uniform fixed
sampling remain unchanged. Exact balanced sampling is deliberately excluded.

## Exact 20x budget

The solver executes `range(num_iterations + 1)`. Experiment 28 therefore runs
81 solve passes from `num_iterations=80`, rather than 80 passes. Experiment 42
sets `num_iterations=1619`, producing exactly `1620 = 20 * 81` solve passes and
exactly 20 times the configured traversal and per-pass regret/value optimizer
budget. The unchanged evaluation cadence also increases average-policy
training checkpoints over the longer horizon. Actual nodes touched remain the
authoritative measure because sampled trajectories can contain different
numbers of nodes; both arms export that count at every checkpoint. Based on
Experiment 28's approximately 0.94 million nodes, the expected endpoint is
roughly 19 million nodes per run.

The experiment uses the sequential backend because the current Ray backend
does not yet implement globally infoset-stratified replay.

## Run

Full two-arm, three-seed experiment:

```bash
python -m experiments.leduc_poker.escher_long_horizon_candidate_ablation.run
```

This is approximately six 20x runs executed sequentially and can take several
days on the Experiment 28 machine type. The runner writes each completed
seed/variant result incrementally, so completed work remains available if a
later run fails.

## Smoke test

The smoke test overrides the long horizon solely to verify both algorithm
paths and output generation:

```bash
python -m experiments.leduc_poker.escher_long_horizon_candidate_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_20x_nodes,policy_q_stratified_uniform_20x_nodes \
  --iterations 2 \
  --traversals 2 \
  --value-traversals 2 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-regret 2 \
  --batch-size-value 2 \
  --batch-size-average-policy 2 \
  --memory-capacity 128 \
  --output-root outputs/smoke_tests
```

A successful smoke test writes paired exploitability, final-window, trajectory
AUC, runtime, replay-composition, and checkpoint outputs for both arms.

## GCP full run

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp42-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_long_horizon_candidate_ablation.run \
    --output-root outputs/cloud/escher-exp42" \
  "n2-standard-4" "432000" "4000" "16000" "100"
```

The principal endpoints are paired final exploitability, the mean over the
final checkpoint window, node-normalised exploitability AUC, and the shape of
the late-training exploitability curve. The analysis should use nodes touched,
not nominal iteration number, when comparing against Experiment 28 or Deep
CFR.
