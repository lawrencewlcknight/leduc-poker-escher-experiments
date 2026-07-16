# Experiment 39: fixed sampling-policy coverage ablation

This experiment tests the ESCHER paper's prediction that convergence worsens
when histories have very small probability under the updating player's fixed
sampling policy. The exact Experiment 28 configuration is the algorithmic
baseline.

The three five-seed arms are:

| Arm | Updating-player fixed sampling policy |
|---|---|
| Experiment 28 baseline | Uniform over legal actions |
| Exact balanced | Probability proportional to descendant terminal-leaf counts |
| Tempered balanced | `0.5 * uniform + 0.5 * exact_balanced` |

The endpoint policies are exact: mixture weight zero is uniform and weight one
is the existing leaf-balanced calculation in `solver.py`. The tempered arm is
a convex mixture, rather than a changing temperature schedule, so it retains
full support and remains fixed throughout a run.

As in the ESCHER algorithm, this fixed policy controls actions sampled for the
updating player. Opponent actions continue to follow the current opponent
strategy. Experiment 28 has `expl=1.0`, so no learned-policy component is mixed
into the updating player's regret-traversal sampling distribution.

All three arms preserve Experiment 28's networks, regret targets, replay,
traversal count, optimizer work, and five paired seeds. Exact tree enumeration
is enabled only to construct the balanced policy and collect small-game
coverage diagnostics; it does not increment the solver's nodes-touched budget.

The implementation verifies that histories sharing an information set imply
the same balanced action probabilities. A mismatch fails explicitly rather
than silently using a history-dependent policy that would violate the clean
fixed-policy comparison.

## Run

Full five-seed run:

```bash
python -m experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run
```

## Smoke test

From the repository root, with the project environment activated, run:

```bash
python -m experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_uniform_fixed_sampling,exact_balanced_fixed_sampling,tempered_balanced_fixed_sampling \
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

A successful smoke test completes all three fixed sampling policies for seed
`1234` and writes a timestamped directory under `outputs/smoke_tests/`
containing paired exploitability results, exact minimum-history reach, observed
infoset coverage, and visit-imbalance curves.

GCP full run:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp39-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run \
    --output-root outputs/cloud/escher-exp39" \
  "n2-standard-4" "86400" "4000" "16000" "100"
```

## Diagnostics

Checkpoint curves and summaries include:

- exact minimum own-policy reach over all enumerated player histories;
- minimum legal-action sampling probability;
- cumulative number of observed infosets;
- minimum, mean, maximum, and coefficient of variation of visits per infoset;
- minimum, mean, and maximum observed own-policy reach;
- exploitability and node-normalised exploitability AUC.

The primary comparison is paired final exploitability and node-normalised AUC
against the exact Experiment 28 uniform-sampling baseline.
