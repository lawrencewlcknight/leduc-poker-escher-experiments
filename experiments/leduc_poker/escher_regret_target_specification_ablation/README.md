# Experiment 35: ESCHER regret-target specification ablation

This experiment tests the paper/code distinction in ESCHER's instantaneous
regret target while carrying forward the complete Experiment 28 candidate
configuration.

The matched-seed variants are:

| Variant | Raw instantaneous regret target |
|---|---|
| Author-code baseline | `Q_hat(h,a) - V_hat(h)` |
| Paper treatment | `Q_hat(h,a) - sum_a pi(a\|h) Q_hat(h,a)` |

Everything else is fixed to Experiment 28, including the `(256, 256, 128)`
plain networks, 64-unit per-action regret head, five development seeds,
traversal/update budgets, replay capacity, and standardized regret-target
processing. This isolates target specification under the current best
configuration.

The solver also exports raw-target diagnostics at every exploitability
checkpoint:

- signed, absolute, and RMS Bellman-consistency residual;
- absolute policy-weighted raw regret target;
- fraction of samples for which every legal raw target is negative;
- number of regret targets represented by each diagnostic checkpoint.

The policy-weighted-Q treatment should have policy-weighted raw target zero up
to floating-point error. Its Bellman residual is still measured because that
quantity describes how far the learned state value is from the policy-weighted
learned child values, even though the treatment no longer inserts the residual
into every action target.

## Run

Default full run:

```bash
python -m experiments.leduc_poker.escher_regret_target_specification_ablation.run
```

## Smoke test

From the repository root, with the project environment activated, run:

```bash
python -m experiments.leduc_poker.escher_regret_target_specification_ablation.run \
  --seeds 1234 \
  --variant-ids author_state_value_baseline,paper_policy_weighted_q \
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

A successful smoke test completes both variants for seed `1234` and writes a
timestamped run directory under `outputs/smoke_tests/` containing
`variant_seed_summary.csv`, `checkpoint_curves.csv`, and the paired baseline
comparison.

GCP full run:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp35-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_specification_ablation.run \
    --output-root outputs/cloud/escher-exp35" \
  "n2-standard-4" "21600" "4000" "16000" "100"
```

The primary outcome is the paired difference in final exploitability. The
node-normalised exploitability AUC determines whether any gain is present over
the whole training trajectory rather than only at the final evaluation.
