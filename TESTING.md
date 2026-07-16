# Testing and Smoke Tests

This file describes the basic checks for the ESCHER Leduc poker repository.

## 1. Environment setup

```bash
python3.9 -m venv venv
source venv/bin/activate            # macOS/Linux
# .\venv\Scripts\Activate.ps1       # Windows PowerShell
pip install --upgrade pip
pip install -r requirements-dev.txt
```

OpenSpiel can require platform-specific installation steps. If `pip install -r requirements.txt` fails on `open_spiel`, follow the official OpenSpiel install guide and rerun.

## 2. Syntax check

```bash
python -m compileall escher_poker experiments tests
```

This catches syntax errors. It does not guarantee that TensorFlow/OpenSpiel are installed correctly.

## 3. Unit tests

The test package covers import checks, thesis-artifact promotion helpers, and
configuration sanity checks for the higher-numbered ESCHER experiments.

```bash
pytest
```

At present, the most important operational test is the smoke run below.

## 4. Quick experiment smoke test

Run a tiny two-seed ESCHER baseline:

```bash
python -m experiments.leduc_poker.escher_multiseed_baseline.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests
```

The run should produce a timestamped output directory containing CSV summaries, JSON metadata, and PNG plots. The smoke-test results should not be interpreted as evidence about ESCHER performance.

Run a tiny one-seed intermediate policy-training ablation:

```bash
python -m experiments.leduc_poker.escher_intermediate_policy_training_ablation.run \
  --seeds 1234 \
  --iterations 10 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce the common output files plus variant and paired-difference summaries.

Run a tiny one-seed checkpoint-stability experiment:

```bash
python -m experiments.leduc_poker.escher_checkpoint_stability.run \
  --seeds 1234 \
  --checkpoint-schedule 1,2 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce checkpoint summaries, policy snapshots, exact head-to-head matrices, and PNG plots.

Run a tiny constrained hyperparameter-search experiment:

```bash
python -m experiments.leduc_poker.escher_constrained_hyperparameter_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 2 \
  --confirmation-iterations 2 \
  --screening-evaluation-interval 1 \
  --confirmation-evaluation-interval 1 \
  --n-random-candidates 1 \
  --confirmation-top-k 1 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --output-root outputs/smoke_tests
```

This should produce screening, confirmation, paired-difference, curve, and plot outputs.

Run a tiny one-seed warm-start fair ablation:

```bash
python -m experiments.leduc_poker.escher_warm_start_fair_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --warm-start-boundary 1 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce paired continuous/warm-start summaries, checkpoint curves, saved checkpoint artifacts, and PNG plots.

Run a tiny one-seed learning-rate schedule ablation:

```bash
python -m experiments.leduc_poker.escher_lr_schedule_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce schedule summaries, paired deltas, checkpoint curves with learning-rate values, and PNG plots.

Run a tiny one-seed reach-weighting ablation:

```bash
python -m experiments.leduc_poker.escher_reach_weighting_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce variant summaries, paired reach-minus-baseline deltas, checkpoint curves, reach diagnostics, and PNG plots.

Run a tiny one-seed value-trajectory reuse ablation:

```bash
python -m experiments.leduc_poker.escher_reuse_value_trajectory_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --value-test-traversals 2 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce variant summaries, paired reuse-minus-baseline deltas, checkpoint curves, traversal-budget diagnostics, and PNG plots.

Run a tiny one-seed disk-backed regret-memory ablation:

```bash
python -m experiments.leduc_poker.escher_disk_backed_regret_memory_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce variant summaries, paired disk-minus-baseline deltas, checkpoint curves, peak-RSS/storage diagnostics, disk replay artifacts, and PNG plots.

Run a tiny one-seed on-policy joint-regret ablation:

```bash
python -m experiments.leduc_poker.escher_on_policy_joint_regret_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

This should produce variant summaries, paired on-policy-minus-baseline deltas, checkpoint curves, traversal-budget diagnostics, and PNG plots.

Run a tiny solver-parameter random search:

```bash
python -m experiments.leduc_poker.escher_solver_parameter_random_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 2 \
  --confirmation-iterations 2 \
  --screening-evaluation-interval 1 \
  --confirmation-evaluation-interval 1 \
  --n-random-candidates 1 \
  --confirmation-top-k 1 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --policy-network-layers 32,32 \
  --regret-network-layers 32,32 \
  --value-network-layers 32,32 \
  --all-actions true \
  --use-balanced-probs false \
  --val-bootstrap false \
  --output-root outputs/smoke_tests
```

This should produce screening and confirmation summaries, paired confirmation deltas, checkpoint curves, sampled-parameter metadata, NPZ export, and PNG plots.

Run a tiny one-seed candidate architecture validation:

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

This should produce the candidate architecture seed summary, checkpoint curves,
aggregate summary, and nodes-touched plots.

Run a tiny matched-arm regret-target specification ablation:

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

This should produce paired paper-target-minus-author-target exploitability,
node-normalised AUC, and raw-target consistency diagnostics.

Run a tiny corrected regret-target scale-only ablation:

```bash
python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run \
  --seeds 1234 \
  --variant-ids experiment_28_batch_centered_baseline,corrected_batch_centered_control,corrected_raw,corrected_fixed_utility_scale,corrected_batch_rms,corrected_persistent_std \
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

This should produce paired comparisons against both the exact Experiment 28
baseline and the corrected batch-centred control, including sign diagnostics.

Run a tiny staged 2x2 correction factorial:

```bash
python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run \
  --screening-seeds 1234 \
  --confirmation-seeds 2025 \
  --confirmation-top-k 1 \
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

This should screen all four cells, rank treatments by node-matched
exploitability, and run the selected treatment plus baseline on a distinct
confirmation seed.

Run a tiny regret replay composition ablation:

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

This should produce paired exploitability outputs plus replay retention,
infoset-balance, and stored-reach curves.

Run a tiny fixed sampling-policy coverage ablation:

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

This should produce paired exploitability outputs plus exact minimum-history
reach, observed infoset coverage, and visit-imbalance curves.

## 5. Full experiment run

```bash
python -m experiments.leduc_poker.escher_multiseed_baseline.run
python -m experiments.leduc_poker.escher_intermediate_policy_training_ablation.run
python -m experiments.leduc_poker.escher_checkpoint_stability.run
python -m experiments.leduc_poker.escher_constrained_hyperparameter_search.run
python -m experiments.leduc_poker.escher_warm_start_fair_ablation.run
python -m experiments.leduc_poker.escher_lr_schedule_ablation.run
python -m experiments.leduc_poker.escher_reach_weighting_ablation.run
python -m experiments.leduc_poker.escher_reuse_value_trajectory_ablation.run
python -m experiments.leduc_poker.escher_disk_backed_regret_memory_ablation.run
python -m experiments.leduc_poker.escher_on_policy_joint_regret_ablation.run
python -m experiments.leduc_poker.escher_solver_parameter_random_search.run
python -m experiments.leduc_poker.escher_diagnostic_hypothesis_sweep.run
python -m experiments.leduc_poker.escher_author_budget_multiseed.run
python -m experiments.leduc_poker.escher_network_size_sweep.run
python -m experiments.leduc_poker.escher_separate_network_architecture_sweep.run
python -m experiments.leduc_poker.escher_regret_network_width_sweep.run
python -m experiments.leduc_poker.escher_policy_network_width_sweep.run
python -m experiments.leduc_poker.escher_layer_norm_ablation.run
python -m experiments.leduc_poker.escher_activation_sweep.run
python -m experiments.leduc_poker.escher_residual_mlp_sweep.run
python -m experiments.leduc_poker.escher_bottleneck_architecture_sweep.run
python -m experiments.leduc_poker.escher_shared_trunk_head_sweep.run
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run
python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run
python -m experiments.leduc_poker.escher_regret_target_specification_ablation.run
python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run
python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run
python -m experiments.leduc_poker.escher_regret_replay_composition_ablation.run
python -m experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run
```

The full runs use the aligned ESCHER baseline configuration. Most experiments default to the 10 thesis seeds; some targeted ablations use the smaller notebook seed set unless `--seeds` is supplied. These may take a long time.
