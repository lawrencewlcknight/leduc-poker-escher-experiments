# Experiment 30: ESCHER candidate learning-rate ablation

This experiment tests whether the Experiment 28 candidate architecture is
sensitive to the constant learning rate.

The baseline is the exact Experiment 28 candidate model:

- plain `(256, 256, 128)` policy, regret, and value trunks;
- no LayerNorm and no residual trunk connections;
- standard linear policy output;
- one 64-unit per-action regret head;
- standardised legal regret targets without clipping;
- revised author-style ESCHER training protocol.

The variants are:

- constant learning rate `1e-3` baseline;
- constant learning rate `5e-4`;
- constant learning rate `2e-3`.

No learning-rate decay schedules are tested in this experiment.

Default full run:

```bash
python -m experiments.leduc_poker.escher_learning_rate_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_learning_rate_ablation.run \
  --seeds 1234 \
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

GCP smoke test:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp30-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_learning_rate_ablation.run \
    --seeds 1234 \
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
    --output-root outputs/cloud/escher-smoke-exp30" \
  "n2-standard-4" "3600" "4000" "16000" "100"
```

The primary comparison is paired final exploitability against the `1e-3`
baseline. Node-normalised exploitability AUC is also exported to compare the
training trajectory as a function of nodes touched.
