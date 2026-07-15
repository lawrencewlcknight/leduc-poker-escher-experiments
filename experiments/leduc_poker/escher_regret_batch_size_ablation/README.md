# Experiment 34: ESCHER regret-batch-size ablation

This experiment follows up the only positive signal in Experiment 31: the
large-replay treatment with regret batch size 512 achieved a slightly lower
mean final exploitability than the 50k replay baseline, but replay capacity and
regret batch size were changed together. This ablation isolates the batch-size
component by keeping replay capacity fixed at the Experiment 28 candidate value
of 50,000 samples.

The baseline is the exact Experiment 28 candidate model:

- plain `(256, 256, 128)` policy, regret, and value trunks;
- no LayerNorm and no residual trunk connections;
- standard linear policy output;
- one 64-unit per-action regret head;
- standardised legal regret targets without clipping;
- revised author-style ESCHER training protocol;
- replay capacity 50,000 with value batch size 256.

The variants are:

| Variant | Replay capacity | Regret batch | Value batch |
|---|---:|---:|---:|
| Baseline | 50,000 | 256 | 256 |
| Regret batch 512 | 50,000 | 512 | 256 |

Default full run:

```bash
python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run \
  --seeds 1234 \
  --variant-ids baseline_regret_batch_256,regret_batch_512 \
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
  "escher-smoke-exp34-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_regret_batch_256 \
    --iterations 2 \
    --traversals 2 \
    --value-traversals 2 \
    --policy-network-train-steps 1 \
    --regret-network-train-steps 1 \
    --value-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp34" \
  "n2-standard-4" "3600" "4000" "16000" "100"
```

The primary comparison is the paired final exploitability of regret batch 512
against the 256-batch baseline. Node-normalised exploitability AUC is also
exported so the effect can be assessed over the full training trajectory rather
than only at the terminal checkpoint.
