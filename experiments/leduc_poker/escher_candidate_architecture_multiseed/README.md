# Experiment 28: ESCHER candidate architecture multi-seed validation

This experiment trains the current best ESCHER candidate architecture over the
five fixed development seeds used in earlier Leduc poker diagnostics.

The candidate combines:

- the Experiment 27 deep plain `(256, 256, 128)` policy, regret, and value
  trunks;
- no LayerNorm and no residual trunk connections;
- the Experiment 22 64-unit per-action regret head;
- the Experiment 23 standardised regret-target treatment, without clipping;
- the revised Experiment 13 training protocol.

Default full run:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run
```

Useful smoke-test settings:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run \
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

The smoke-test output checks only that the experiment entry point, candidate
configuration, plotting, and summary exports are operational. It is not an
estimate of ESCHER performance.

