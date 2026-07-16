# Experiment 41: combined ESCHER candidate ablation

This experiment tests whether the modest improvements identified in
Experiments 35–39 combine into a stronger Leduc poker configuration. It runs
three matched configurations over seeds `1234`, `2025`, and `31415`:

| Arm | Regret target | Target processing | Regret replay | Fixed sampling |
|---|---|---|---|---|
| Experiment 28 baseline | Author state value | Batch-centred standardization | Global reservoir | Uniform actions |
| Evidence-weighted candidate | Policy-weighted Q | Batch-centred standardization | Infoset-stratified | Uniform actions |
| Maximum stack | Policy-weighted Q | Batch-centred standardization | Infoset-stratified | Exact leaf-balanced |

Everything not shown in the table is inherited exactly from Experiment 28,
including its architecture, traversal counts, replay capacity, optimizer work,
learning rate, evaluation schedule, and network reinitialization settings.
The experiment uses the sequential solver because the current Ray backend does
not yet support globally infoset-stratified replay.

The primary paired comparisons are:

- each treatment minus the exact Experiment 28 baseline;
- maximum stack minus the evidence-weighted uniform-sampling candidate.

The second comparison isolates the incremental effect of exact balanced
sampling after adopting policy-weighted Q targets and stratified replay.

## Run

Full three-seed experiment from the repository root:

```bash
python -m experiments.leduc_poker.escher_combined_candidate_ablation.run
```

## Smoke test

With the project environment activated, run:

```bash
python -m experiments.leduc_poker.escher_combined_candidate_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_baseline,policy_q_stratified_uniform,policy_q_stratified_exact_balanced \
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

A successful smoke test completes all three arms and writes a timestamped
directory containing `variant_seed_summary.csv`,
`paired_differences_vs_baseline.csv`,
`paired_differences_vs_policy_q_stratified_uniform.csv`, aggregate summaries,
checkpoint curves, and exploitability/replay-composition plots.

## GCP full run

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp41-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_combined_candidate_ablation.run \
    --output-root outputs/cloud/escher-exp41" \
  "n2-standard-4" "86400" "4000" "16000" "100"
```

The principal quality measures are paired final exploitability,
node-normalised exploitability AUC, final-window mean exploitability, and the
fraction of seeds improved. Replay samples-per-infoset imbalance is exported
to verify that the stratified treatments produce their intended data
composition.
