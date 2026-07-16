# Experiment 36: corrected regret-target scale-only processing

This matched-seed experiment tests whether Experiment 28's minibatch mean
subtraction improves neural fitting while changing ESCHER's game-solving
dynamics. The primary baseline is the exact Experiment 28 configuration.

The six arms are:

| Variant | Target baseline | Supervised-target processing |
|---|---|---|
| Experiment 28 baseline | Author state value | Batch mean subtraction and std scaling |
| Corrected batch-centred control | Policy-weighted child Q | Batch mean subtraction and std scaling |
| Corrected raw | Policy-weighted child Q | None |
| Corrected fixed utility scale | Policy-weighted child Q | Divide by fixed Leduc utility range, 26 |
| Corrected minibatch RMS | Policy-weighted child Q | Divide by minibatch RMS only |
| Corrected persistent std | Policy-weighted child Q | Divide by EMA std only |

Every arm otherwise inherits Experiment 28, including its five seeds,
architecture, traversal and update budgets, replay settings, optimizer, and
evaluation schedule. [OpenSpiel reports Leduc utilities in `[-13, 13]`](https://openspiel.readthedocs.io/en/latest/api_reference/game_max_min_utility.html),
so the fixed game-wide regret scale is the complete utility range, `26`.

Two paired comparisons are exported:

- every arm minus the exact Experiment 28 baseline;
- every corrected arm minus the corrected batch-centred control.

The second comparison isolates target processing after adopting the
policy-weighted regret correction. It avoids interpreting a combined
target-definition and target-processing change as a normalization result.

At each exploitability checkpoint the experiment records the most recent
training minibatch's applied mean and scale. Sign-flip and raw/processed
positive-target fractions are averaged over the latest regret-network training
event. Positive-only scale modes should have exactly zero sign flips;
batch-centred standardization need not.

## Run

Full run:

```bash
python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run
```

## Smoke test

From the repository root, with the project environment activated, run:

```bash
python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_batch_centered_baseline,corrected_batch_centered_control,corrected_raw,corrected_fixed_utility_scale,corrected_batch_rms,corrected_persistent_std \
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

A successful smoke test completes all six arms for seed `1234` and writes a
timestamped directory under `outputs/smoke_tests/`. Check
`paired_differences_vs_baseline.csv` and
`paired_differences_vs_corrected_batch_centered_control.csv` to confirm both
planned contrasts were produced.

Primary outputs include `paired_differences_vs_baseline.csv` and
`paired_differences_vs_corrected_batch_centered_control.csv`, as well as the
usual seed summaries, checkpoint curves, aggregate JSON, and plots.

GCP full run:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp36-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run \
    --output-root outputs/cloud/escher-exp36" \
  "n2-standard-4" "86400" "4000" "16000" "100"
```
