# ESCHER Action-Head LayerNorm/Residual-LN Ablation

This experiment tests whether the Deep CFR Experiment 13 layer-normalisation
network hypothesis transfers to the carried-forward ESCHER action-head model.

The baseline is the confirmed carried-forward ESCHER configuration:

- Experiment 13 training protocol: 80 iterations, 500 regret traversals, and
  500 history-value traversals per iteration.
- No importance-sampling correction and uniform zero-regret fallback.
- Policy, regret, and value trunks with hidden layers `(256, 128)`.
- Standard linear policy output.
- Regret networks with one 64-unit separate per-action head.

The variants isolate trunk normalisation and residual structure while keeping
the policy output and regret action heads fixed:

- `carry_forward_layer_norm_256_128_action_heads`: current carried-forward
  LayerNorm baseline.
- `plain_256_128_action_heads`: same trunk sizes with LayerNorm removed.
- `deep_plain_256_256_128_action_heads`: deeper plain trunk.
- `deep_layer_norm_256_256_128_action_heads`: deeper LayerNorm trunk.
- `deep_residual_layer_norm_256_256_128_action_heads`: deeper LayerNorm trunk
  with a same-width residual connection on the repeated 256-unit layer.

## Run

```bash
python -m experiments.leduc_poker.escher_action_head_layer_norm_residual_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_action_head_layer_norm_residual_ablation.run \
  --iterations 4 \
  --evaluation-interval 2 \
  --traversals 20 \
  --value-traversals 20 \
  --policy-network-train-steps 10 \
  --regret-network-train-steps 10 \
  --value-network-train-steps 10 \
  --output-root outputs/smoke_tests
```
