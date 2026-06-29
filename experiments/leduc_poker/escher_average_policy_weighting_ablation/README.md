# ESCHER Average-Policy Weighting Ablation

This experiment tests whether the Deep CFR Experiment 10 average-strategy
weighting hypothesis transfers to ESCHER.

The baseline is the carried-forward ESCHER configuration:

- Experiment 13 training protocol: 80 iterations, 500 regret traversals, and
  500 history-value traversals per iteration.
- No importance-sampling correction and uniform zero-regret fallback.
- Policy, regret, and value trunks with hidden layers `(256, 128)`.
- Standard linear policy output.
- Regret networks with one 64-unit separate per-action head.

Two variants are compared under matched seeds:

- `linear_avg_weighting_baseline`: the current ESCHER behaviour, where
  average-policy samples are weighted by CFR iteration.
- `uniform_avg_weighting`: gives each sampled average-policy memory equal
  weight in the supervised average-policy loss.

The main output is a paired multi-seed comparison of final exploitability,
best intermediate exploitability, exploitability AUC against nodes touched, and
policy-value error.

## Run

```bash
python -m experiments.leduc_poker.escher_average_policy_weighting_ablation.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_average_policy_weighting_ablation.run \
  --seeds 1234 \
  --iterations 4 \
  --evaluation-interval 2 \
  --traversals 20 \
  --value-traversals 20 \
  --policy-network-train-steps 10 \
  --regret-network-train-steps 10 \
  --value-network-train-steps 10 \
  --output-root outputs/smoke_tests
```
