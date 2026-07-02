# Leduc Poker ESCHER Regret-Target Processing Ablation

This experiment tests whether simple preprocessing of sampled ESCHER regret
targets improves Leduc poker performance. Replay buffers store raw sampled
regret targets in every variant; processing is applied only to the supervised
regret-network fitting loss. Processing is applied to legal-action regret
entries, after which illegal-action targets remain masked to zero.

The baseline is the carried-forward ESCHER configuration:

- 80 ESCHER iterations
- 500 regret traversals and 500 history-value traversals per iteration
- no importance-sampling correction in regret targets
- uniform legal-action fallback when all positive regrets are zero
- `(256, 128)` policy, regret, and value trunks
- one 64-unit separate per-action head on the regret networks only
- standard linear output head for the policy network

The default variants are:

| Variant id | Processing |
| --- | --- |
| `raw_regret_targets` | Raw carried-forward ESCHER baseline. |
| `standardized_regret_targets` | Batch-standardized legal regret targets. |
| `clipped_regret_targets` | Legal regret targets clipped to `[-1.0, 1.0]`. |
| `standardized_clipped_regret_targets` | Legal regret targets batch-standardized, then clipped to `[-1.0, 1.0]`. |

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.escher_regret_target_processing_ablation.run
```

By default, each seed/variant arm is run sequentially in a fresh Python worker
process. This keeps the command form unchanged while releasing TensorFlow
solver, network, and replay-memory state between full ESCHER trainings. For
local debugging only, this can be disabled:

```bash
python -m experiments.leduc_poker.escher_regret_target_processing_ablation.run \
  --disable-subprocess-isolation
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.escher_regret_target_processing_ablation.run \
  --seeds 1234 \
  --variant-ids raw_regret_targets,standardized_clipped_regret_targets \
  --iterations 2 \
  --traversals 2 \
  --value-traversals 2 \
  --evaluation-interval 1 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --batch-size-regret 2 \
  --batch-size-value 2 \
  --batch-size-average-policy 2 \
  --memory-capacity 128 \
  --output-root outputs/smoke_tests
```

Useful CLI options:

- `--seeds` selects a comma-separated seed list.
- `--variant-ids` selects a comma-separated subset of processing arms.
- `--regret-target-clip-value` changes the clipping threshold.
- `--regret-target-standardize-epsilon` changes the minimum standardisation
  scale.
- `--save-final-checkpoints true` writes final full-model checkpoints.

## Outputs

- `variant_seed_summary.csv` records one row per variant and seed.
- `checkpoint_curves.csv` records intermediate exploitability, value, loss,
  processed-target variance, standardisation scale, and clip-fraction
  diagnostics.
- `aggregate_summary.json` aggregates metrics by target-processing variant.
- `paired_differences_vs_baseline.csv` reports per-seed treatment minus raw
  baseline deltas.
- `paired_difference_summary.json` aggregates paired differences and
  improvement fractions.
- `worker_results/` and `worker_logs/` contain one isolated worker output and
  log per seed/variant arm when subprocess isolation is enabled.
- `partial_variant_seed_summary.jsonl` and `partial_checkpoint_curves.jsonl`
  are appended after each completed arm.
- PNG plots include final exploitability, exploitability by nodes touched,
  average-policy value by nodes touched, processed target variance, and
  clipping fraction.

Paired differences are reported as `variant - raw_regret_targets`; negative
exploitability, AUC, and policy-value-error deltas are improvements.
