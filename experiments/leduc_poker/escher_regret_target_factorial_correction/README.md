# Experiment 37: 2x2 regret-target correction factorial

This experiment is the priority screening test for two proposed corrections to
Experiment 28:

1. replace the authors' learned state-value baseline with the policy-weighted
   child-Q baseline;
2. replace batch-centred standardization with scale-only minibatch RMS
   normalization.

The four screening arms form a complete 2x2 factorial:

| Arm | Regret baseline | Target processing |
|---|---|---|
| Current baseline and standardization | Author state value | Batch mean subtraction and std scaling |
| Policy-weighted Q baseline only | Policy-weighted child Q | Batch mean subtraction and std scaling |
| Scale-only normalization only | Author state value | Divide by minibatch RMS only |
| Both corrections | Policy-weighted child Q | Divide by minibatch RMS only |

All other settings are inherited exactly from Experiment 28. This includes its
architecture, traversal and training budgets, replay capacity, optimizer,
evaluation interval, and zero-regret fallback.

## Staged design

Screening runs all four arms on three seeds: `1234`, `2025`, and `31415`. The
runner ranks treatments by mean exploitability at approximately one million
nodes. It then confirms the two best treatment arms and the exact Experiment 28
baseline on five separate seeds: `27182`, `16180`, `4242`, `8675309`, and `7`.

The confirmation seeds do not overlap the screening seeds. This avoids treating
a larger rerun of the selection data as independent confirmation.

The primary metric is linearly interpolated at one million nodes when the curve
brackets that budget. If a run stops just short of one million nodes, the
nearest endpoint is used without extrapolation and the node gap is recorded.
Exploitability strictly below `0.3` is marked as a meaningful success.

Screening also exports paired factorial effects for each seed:

- policy-weighted-Q main effect;
- scale-only-normalization main effect;
- correction interaction, calculated as a difference in differences.

Negative effects improve exploitability.

## Run

Full staged run:

```bash
python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run
```

## Smoke test

From the repository root, with the project environment activated, run this
small end-to-end test of both the screening and confirmation stages:

```bash
python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run \
  --screening-seeds 1234 \
  --confirmation-seeds 2025 \
  --confirmation-top-k 1 \
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

A successful smoke test writes a timestamped directory under
`outputs/smoke_tests/` with completed `screening/` and `confirmation/`
subdirectories, including `screening_ranking.csv` and the factorial-effect
summary.

GCP full run:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp37-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run \
    --output-root outputs/cloud/escher-exp37" \
  "n2-standard-4" "86400" "4000" "16000" "100"
```

## Principal outputs

The `screening/` directory contains arm-level summaries, checkpoint curves,
paired baseline differences, `screening_ranking.csv`,
`factorial_effects_by_seed.csv`, and `factorial_effect_summary.json`.

The `confirmation/` directory contains the corresponding outputs for the exact
baseline and selected treatments. Both stages include a node-matched bar chart
with the `0.3` threshold and exploitability curves by nodes touched.
