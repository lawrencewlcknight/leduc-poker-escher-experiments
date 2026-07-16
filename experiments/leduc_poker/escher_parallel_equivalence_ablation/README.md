# Experiment 40: sequential versus Ray-parallel ESCHER

This experiment compares the Experiment 28 sequential implementation with a
Ray-parallel experience-collection implementation over the paired seeds
`1234`, `2025`, and `31415`. The parallel implementation is adapted from
Sandholm-Lab's `parallelized_ESCHER.py` at commit
`e694eaaa251952696aaf36ef1c790887c8324750`; attribution and licensing are in
the repository's `THIRD_PARTY_NOTICES.md`.

Both arms use the same Experiment 28 network architecture, regret target,
optimizer work, total traversal counts, and total replay capacity. The
parallel arm has one central learner and three CPU Ray actors. Every traversal
budget is split exactly across the actors, and each replay capacity is split
so the sum of actor capacities equals the sequential capacity. This avoids the
upstream implementation's implicit multiplication of replay capacity by the
worker count.

## What “the same answer” means

Parallel random sampling uses independent deterministic worker streams, so
bitwise-identical weights and exploitability are neither expected nor required.
The primary test is practical equivalence of the final policies:

- final exploitability absolute difference no greater than `0.05`;
- final player-0 policy-value absolute difference no greater than `0.02`.

The output records the per-seed checks and a paired TOST-style test: the 90%
confidence interval for the mean parallel-minus-sequential difference must lie
strictly inside the pre-declared margin. Three seeds make this an equivalence
screen, not a high-powered proof; a five-or-more-seed confirmation is warranted
if a conference claim rests on the result.

## Runtime measures

The experiment reports sequential-time divided by parallel-time, so speedup
greater than one favours parallel execution, for:

- solver/Ray initialization;
- the complete training loop;
- end-to-end initialization plus training;
- regret, value, and combined experience collection.

Ray startup and experience transfer can dominate a small Leduc workload. A
Leduc end-to-end slowdown would not invalidate the backend for larger games;
the experience-collection-phase speedup is the cleanest scaling signal. That
phase includes learner-to-worker weight synchronization and Ray orchestration,
while the full-loop measure also includes replay transfer and learner updates.

Worker random streams are reproducible: zero-based worker `k` uses
`run_seed + 1_000_003 * (k + 1)`.

## Run

Full three-seed comparison from the repository root:

```bash
python -m experiments.leduc_poker.escher_parallel_equivalence_ablation.run
```

The default three-worker arm is intended for a machine with at least four CPU
cores, leaving one core available to the central learner and driver.

## Smoke test

With the project environment activated, run:

```bash
python -m experiments.leduc_poker.escher_parallel_equivalence_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_sequential,experiment_28_ray_parallel \
  --iterations 2 \
  --traversals 6 \
  --value-traversals 6 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-regret 2 \
  --batch-size-value 2 \
  --batch-size-average-policy 2 \
  --memory-capacity 129 \
  --output-root outputs/smoke_tests
```

A successful smoke test completes both arms and writes a timestamped output
directory containing `variant_seed_summary.csv`,
`paired_differences_vs_baseline.csv`,
`paired_difference_summary.json`, and the timing/quality plots. The deliberately
non-divisible capacity `129` also exercises exact three-way budget partitioning.

## GCP full run

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp40-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_parallel_equivalence_ablation.run \
    --output-root outputs/cloud/escher-exp40" \
  "n2-standard-4" "86400" "4000" "16000" "100"
```

The most important output fields are `final_exploitability`,
`final_policy_value`, `experience_collection_speedup`, `end_to_end_speedup`,
and the two entries under `equivalence` in
`paired_difference_summary.json`.
