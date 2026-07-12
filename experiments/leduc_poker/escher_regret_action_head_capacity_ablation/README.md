# Experiment 33: ESCHER regret action-head capacity ablation

This experiment tests whether the Experiment 28 candidate architecture is
limited by the capacity of the per-action regret head. The shared trunk, value
network, average-policy network, replay settings, target processing, and
training protocol are held fixed.

The variants are:

| Variant | Regret action-head depth | Regret action-head units |
|---|---:|---:|
| Baseline | 1 | 64 |
| Wider head | 1 | 128 |
| Deeper head | 2 | 64 |

Default full run:

```bash
python -m experiments.leduc_poker.escher_regret_action_head_capacity_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_regret_action_head_capacity_ablation.run \
  --seeds 1234 \
  --variant-ids baseline_regret_head_64 \
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
  "escher-smoke-exp33-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_action_head_capacity_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_regret_head_64 \
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
    --output-root outputs/cloud/escher-smoke-exp33" \
  "n2-standard-4" "3600" "4000" "16000" "100"
```

The primary comparison is paired final exploitability against the 64-unit
Experiment 28 regret head. The node-based exploitability curve is exported to
test whether additional head capacity delays or breaks the observed plateau.
