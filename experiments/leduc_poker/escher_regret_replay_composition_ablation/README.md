# Experiment 38: regret replay composition ablation

This experiment tests whether Experiment 28's exploitability plateau is partly
caused by finite regret replay rather than the ESCHER update itself. The exact
Experiment 28 configuration is the baseline.

The five matched-seed arms are:

| Arm | Regret replay behavior |
|---|---|
| Experiment 28 reservoir | Global uniform reservoir capped at `50,000` samples per player |
| Store every sample | Append-only, uncapped regret replay |
| Infoset-stratified | Equal-capacity per-infoset reservoirs within the same `50,000` cap |
| Rare-history quotas | Up to 64 protected samples per infoset plus a global overflow reservoir |
| Counterfactual-reach weighted | Weighted priority reservoir using opponent-and-chance reach |

Only regret replay changes. Value replay and average-policy replay retain the
Experiment 28 reservoir implementation and capacity. This isolates the replay
approximation most directly connected to regret-network fitting.

Optimizer work is also held exactly fixed at Experiment 28's setting: 200
regret-network steps per player and iteration, with unchanged batch size. The
all-samples arm therefore tests the predicted reduction in sample reuse rather
than silently compensating with additional optimization.

In-memory regret training shuffles over the full stored population. This is
identical to the existing behavior for Experiment 28's `50,000`-sample buffer,
which is smaller than the standard `100,000`-sample shuffle window, and avoids
an insertion-order bias once the uncapped arm grows beyond that window.

Infosets are identified from the exact information-state tensor used by the
regret network. Counterfactual reach is the opponent-and-chance reach already
tracked by the standard ESCHER traversal. A weighted reservoir is used instead
of adding reach to the regret loss, so the experimental factor remains replay
composition.

## Run

Full five-seed run:

```bash
python -m experiments.leduc_poker.escher_regret_replay_composition_ablation.run
```

## Smoke test

From the repository root, with the project environment activated, run:

```bash
python -m experiments.leduc_poker.escher_regret_replay_composition_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_reservoir_replay,all_regret_samples,infoset_stratified_replay,rare_history_quota_replay,counterfactual_reach_weighted_replay \
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

A successful smoke test completes all five replay modes for seed `1234` and
writes a timestamped directory under `outputs/smoke_tests/` containing paired
exploitability results and the replay-retention, infoset-balance, and
stored-reach curves.

GCP full run:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp38-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_replay_composition_ablation.run \
    --output-root outputs/cloud/escher-exp38" \
  "n2-standard-4" "86400" "4000" "16000" "100"
```

## Diagnostics

Checkpoint curves and final summaries include:

- regret stream size and retained fraction;
- unique stored infosets;
- minimum, mean, maximum, and coefficient of variation of samples per infoset;
- mean stored counterfactual reach for the weighted arm;
- unchanged optimizer-step configuration and peak memory use.

The primary result remains paired final exploitability and node-normalised
exploitability AUC versus the exact Experiment 28 reservoir baseline.
