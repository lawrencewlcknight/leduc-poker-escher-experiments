# Experiment 31: ESCHER replay-capacity ablation

This experiment tests whether the Experiment 28 candidate architecture plateaus
because replay buffers are too small, or because regret-network fitting needs a
larger batch once replay capacity is increased.

The baseline is the exact Experiment 28 candidate model:

- plain `(256, 256, 128)` policy, regret, and value trunks;
- no LayerNorm and no residual trunk connections;
- standard linear policy output;
- one 64-unit per-action regret head;
- standardised legal regret targets without clipping;
- revised author-style ESCHER training protocol.

The variants are:

| Variant | Replay capacity | Regret batch | Value batch |
|---|---:|---:|---:|
| Baseline | 50,000 | 256 | 256 |
| Medium replay | 100,000 | 256 | 256 |
| Large replay | 200,000 | 256 | 256 |
| Large replay + regret batch | 200,000 | 512 | 256 |

Default full run:

```bash
python -m experiments.leduc_poker.escher_replay_capacity_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_replay_capacity_ablation.run \
  --seeds 1234 \
  --variant-ids baseline_replay_50k \
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
  "escher-smoke-exp31-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_replay_capacity_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_replay_50k \
    --iterations 2 \
    --traversals 2 \
    --value-traversals 2 \
    --policy-network-train-steps 1 \
    --regret-network-train-steps 1 \
    --value-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp31" \
  "n2-standard-4" "3600" "4000" "16000" "100"
```

The primary comparison is paired final exploitability against the 50k replay
baseline. Node-normalised exploitability AUC is also exported to compare the
training trajectory as a function of nodes touched.
