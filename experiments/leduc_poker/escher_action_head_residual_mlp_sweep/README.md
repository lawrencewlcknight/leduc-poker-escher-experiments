# ESCHER Action-Head Residual-MLP Sweep

This experiment tests whether residual trunk structure helps after the current
carried-forward ESCHER architecture has already added regret-network action
heads. It should be read as a focused follow-up to the earlier residual-MLP
sweep, not as a replacement for it.

The baseline is the carried-forward ESCHER configuration:

- 80 ESCHER iterations
- 500 regret traversals and 500 history-value traversals per iteration
- no importance-sampling correction in regret targets
- uniform legal-action fallback when all positive regrets are zero
- `(256, 128)` policy, regret, and value trunks
- standard linear output head for the policy network
- one 64-unit separate per-action head on the regret networks

The default variants are:

| Variant id | Description |
| --- | --- |
| `carry_forward_256_128_action_heads` | Current carried-forward baseline. |
| `deep_plain_256_256_128_action_heads` | Deeper plain trunk with residuals disabled. |
| `deep_same_width_256_256_128_action_heads` | Same trunk as the deep plain variant, with a same-width residual skip. |
| `bottleneck_plain_256_128_128_action_heads` | Bottleneck plain trunk with residuals disabled. |
| `bottleneck_projection_256_128_128_action_heads` | Same bottleneck trunk, with a projection residual skip on the width-changing layer. |

All variants keep the regret action-head architecture fixed, so the treatment
variable is the residual structure of the policy, regret, and history-value
trunks.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.escher_action_head_residual_mlp_sweep.run
```

Quick smoke test:

```bash
python -m experiments.leduc_poker.escher_action_head_residual_mlp_sweep.run \
  --seed 1234 \
  --variant-ids carry_forward_256_128_action_heads,deep_same_width_256_256_128_action_heads \
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
  --output-root outputs/smoke_tests
```

## Outputs

The run uses the shared one-seed ESCHER variant runner and writes:

- `variant_summary.csv`
- `checkpoint_curves.csv`
- `summary.json`
- `experiment_metadata.json`
- `final_exploitability_by_variant.png`
- `intermediate_exploitability_by_iteration.png`
- `average_policy_value_by_iteration.png`

For thesis figures, prefer either endpoint comparisons or regenerate
longitudinal curves from `checkpoint_curves.csv` with nodes touched on the
x-axis.
