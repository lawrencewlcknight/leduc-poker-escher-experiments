# ESCHER Factorised Regret-Output Head Ablation

This experiment tests whether the Deep CFR factorised advantage-head idea
transfers to ESCHER's regret networks.

The baseline is the carried-forward ESCHER configuration:

- Experiment 13 training protocol: 80 iterations, 500 regret traversals, and
  500 history-value traversals per iteration.
- No importance-sampling correction and uniform zero-regret fallback.
- Policy, regret, and value trunks with hidden layers `(256, 128)`.
- Standard linear policy output.
- Regret networks with one 64-unit separate per-action head.

The variants isolate the regret-output factorisation:

- `direct_regret_action_head_64_baseline`: the carried-forward direct
  per-action regret heads.
- `centered_regret_action_head_64`: legal-action regret outputs are centred to
  zero mean for each information state.
- `dueling_regret_action_head_64`: a scalar state-value head is added to
  centred legal-action regret deviations.

The centred and dueling variants centre over legal actions only; illegal-action
outputs remain zero after masking.

## Run

```bash
python -m experiments.leduc_poker.escher_factorised_regret_head_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_factorised_regret_head_ablation.run \
  --iterations 4 \
  --evaluation-interval 2 \
  --traversals 20 \
  --value-traversals 20 \
  --policy-network-train-steps 10 \
  --regret-network-train-steps 10 \
  --value-network-train-steps 10 \
  --output-root outputs/smoke_tests
```
