# Experiment 32: ESCHER regret/value work-balance ablation

This experiment tests whether the Experiment 28 candidate architecture is
limited by insufficient regret learning rather than by the trunk architecture
itself. The model, replay settings, regret-target processing, and learning rate
are held fixed; only the allocation of traversal and supervised fitting effort
between the regret and history-value pathways changes.

The variants are:

| Variant | Regret traversals | Value traversals | Regret steps | Value steps |
|---|---:|---:|---:|---:|
| Baseline | 500 | 500 | 200 | 200 |
| Regret-data heavy | 625 | 250 | 200 | 200 |
| Regret-update heavy | 500 | 500 | 300 | 100 |
| Regret-data + update heavy | 625 | 250 | 300 | 100 |

The regret-data-heavy arm keeps the nominal traversal budget approximately
matched because ESCHER collects regret traversals for both players. The
regret-update-heavy arm keeps the combined regret/value supervised update budget
matched.

Default full run:

```bash
python -m experiments.leduc_poker.escher_regret_value_work_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_regret_value_work_ablation.run \
  --seeds 1234 \
  --variant-ids baseline_regret_value_work \
  --iterations 2 \
  --traversals 2 \
  --value-traversals 2 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

GCP smoke test:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp32-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_value_work_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_regret_value_work \
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
    --output-root outputs/cloud/escher-smoke-exp32" \
  "n2-standard-4" "3600" "4000" "16000" "100"
```

The primary comparison is paired final exploitability against the Experiment 28
work allocation. Node-normalised exploitability AUC is exported to distinguish
lower final exploitability from a stronger learning trajectory.
