# Leduc Poker ESCHER Experiments

This repository contains reproducible experiments for evaluating ESCHER-style neural counterfactual regret minimisation on Leduc poker using DeepMind's OpenSpiel library.

The immediate aim is to establish a thesis-quality ESCHER baseline that is aligned with the sister Deep CFR repository. Leduc poker is used as the diagnostic environment because it is a small two-player zero-sum imperfect-information game with a known game value and exact exploitability evaluation. The results from this repository are intended to sit alongside the Deep CFR Leduc poker experiments in an MPhil thesis on neural CFR methods for poker.

The repository is organised so that each experiment can be run independently while sharing reusable ESCHER code. The shared `escher_poker` package contains the ESCHER solver, neural-network definitions, reservoir replay buffer, plotting helpers, seeding utilities, and experiment export utilities. Each experiment lives in its own package under `experiments/leduc_poker/<experiment_name>/`.

## Repository structure

```text
.
├── escher_poker/                                      # Shared reusable code
│   ├── solver.py                                      # ESCHER solver
│   ├── networks.py                                    # Policy, regret, and history-value networks
│   ├── replay.py                                      # Reservoir replay buffer
│   ├── experiment_utils.py                            # Run-dir, metric, and export helpers
│   ├── plotting.py                                    # Thesis-style plots
│   ├── ablation_plotting.py                           # Multi-arm ablation plots
│   ├── policy_snapshots.py                            # Saved policy snapshot helpers
│   ├── checkpoint_analysis.py                         # Exact checkpoint head-to-head analysis
│   ├── checkpoint_plotting.py                         # Checkpoint-stability plots
│   ├── hyperparameter_search.py                       # Search-stage runners and summaries
│   ├── constants.py                                   # Leduc value, thresholds, shuffle sizes
│   └── seeding.py                                     # TensorFlow/NumPy/Python seeding helpers
├── experiments/
│   └── leduc_poker/
│       ├── escher_multiseed_baseline/                 # Experiment 1
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_intermediate_policy_training_ablation/ # Experiment 2
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_checkpoint_stability/               # Experiment 3
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_constrained_hyperparameter_search/  # Experiment 4
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_warm_start_fair_ablation/           # Experiment 5
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_lr_schedule_ablation/               # Experiment 6
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_reach_weighting_ablation/           # Experiment 7
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_reuse_value_trajectory_ablation/    # Experiment 8
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_disk_backed_regret_memory_ablation/ # Experiment 9
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_on_policy_joint_regret_ablation/   # Experiment 10
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_solver_parameter_random_search/    # Experiment 11
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_diagnostic_hypothesis_sweep/       # Experiment 12
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_author_budget_multiseed/           # Experiment 13
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_network_size_sweep/                # Experiment 14
│       ├── escher_separate_network_architecture_sweep/ # Experiment 15
│       ├── escher_regret_network_width_sweep/        # Experiment 16
│       ├── escher_policy_network_width_sweep/        # Experiment 17
│       ├── escher_layer_norm_ablation/               # Experiment 18
│       ├── escher_activation_sweep/                  # Experiment 19
│       ├── escher_residual_mlp_sweep/                # Experiment 20
│       ├── escher_bottleneck_architecture_sweep/     # Experiment 21
│       ├── escher_shared_trunk_head_sweep/           # Experiment 22
│       ├── escher_regret_target_processing_ablation/ # Experiment 23
│       ├── escher_action_head_residual_mlp_sweep/    # Experiment 24
│       ├── escher_average_policy_weighting_ablation/ # Experiment 25
│       ├── escher_factorised_regret_head_ablation/   # Experiment 26
│       ├── escher_action_head_layer_norm_residual_ablation/ # Experiment 27
│       ├── escher_candidate_architecture_multiseed/  # Experiment 28
│       ├── escher_reinitialisation_ablation/         # Experiment 29
│       ├── escher_learning_rate_ablation/            # Experiment 30
│       ├── escher_replay_capacity_ablation/          # Experiment 31
│       ├── escher_regret_value_work_ablation/        # Experiment 32
│       ├── escher_regret_action_head_capacity_ablation/ # Experiment 33
│       ├── escher_regret_batch_size_ablation/        # Experiment 34
│       ├── escher_regret_target_specification_ablation/ # Experiment 35
│       ├── escher_regret_target_scale_only_ablation/ # Experiment 36
│       ├── escher_regret_target_factorial_correction/ # Experiment 37
│       ├── escher_regret_replay_composition_ablation/ # Experiment 38
│       ├── escher_fixed_sampling_coverage_ablation/ # Experiment 39
│       ├── escher_parallel_equivalence_ablation/    # Experiment 40
│       ├── escher_combined_candidate_ablation/      # Experiment 41
│       └── escher_long_horizon_candidate_ablation/  # Experiment 42
├── docs/
│   └── OUTPUT_CONVENTIONS.md
├── notebooks/                                        # Original notebook archive
├── outputs/                                          # Experiment outputs (gitignored)
├── tests/                                            # Import, config, and artifact-helper tests
├── venv/                                             # Placeholder only; environment not committed
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── THIRD_PARTY_NOTICES.md
└── TESTING.md
```

## Experiments

### 1. Leduc poker ESCHER multi-seed baseline

[`experiments/leduc_poker/escher_multiseed_baseline/`](experiments/leduc_poker/escher_multiseed_baseline/README.md)

Runs the aligned ESCHER baseline on OpenSpiel `leduc_poker` across the same ten random seeds used in the Deep CFR baseline experiments. The default solver specification is deliberately lightweight for Leduc poker, using smaller 64-by-64 networks, fewer traversals, fewer supervised training steps, and less frequent intermediate exploitability checks than the original notebook-aligned configuration. The primary metric is exploitability, reported as NashConv divided by two. Secondary metrics include policy-value error from the known Leduc game value, nodes touched, wall-clock time, and final/best/final-window exploitability. Diagnostic metrics include average-policy loss, regret-network losses, history-value-network train/test loss, and replay-buffer sizes.

**Question:** under a fixed training protocol, does the ESCHER implementation learn a low-exploitability average policy in Leduc poker, and how variable is the result across random seeds?

### 2. ESCHER intermediate average-policy training ablation

[`experiments/leduc_poker/escher_intermediate_policy_training_ablation/`](experiments/leduc_poker/escher_intermediate_policy_training_ablation/README.md)

Compares the baseline ESCHER diagnostic protocol against final-only average-policy extraction. In the baseline, each intermediate exploitability checkpoint trains a playable average-policy network from average-policy memory. The ablation asks whether those repeated supervised policy-network training events affect final exploitability, or whether they are mainly an evaluation cost.

**Question:** does disabling intermediate policy-network training change final ESCHER performance, once the regret/history-value training configuration and seeds are held fixed?

The experiment has three arms: the baseline intermediate-checkpoint regime, final-only policy training with the usual single-event budget, and final-only policy training with the baseline's total policy-gradient budget matched at final extraction.

### 3. ESCHER checkpoint-stability head-to-head experiment

[`experiments/leduc_poker/escher_checkpoint_stability/`](experiments/leduc_poker/escher_checkpoint_stability/README.md)

Saves playable average-policy checkpoints during ESCHER training and evaluates whether later checkpoints consistently beat earlier checkpoints in exact head-to-head play. The experiment also supports a continuous-baseline arm so the checkpoint/resume mechanism can be checked against a single uninterrupted ESCHER run.

**Question:** as ESCHER training progresses, do later average-policy checkpoints become stronger than earlier checkpoints, or is checkpoint quality non-monotonic?

### 4. ESCHER constrained hyperparameter search

[`experiments/leduc_poker/escher_constrained_hyperparameter_search/`](experiments/leduc_poker/escher_constrained_hyperparameter_search/README.md)

Runs a bounded search around the ESCHER baseline configuration to test whether poor convergence in Leduc poker can be explained by avoidable optimisation or approximation settings. The experiment uses a screening stage over baseline, targeted, and random candidates, then confirms the strongest candidates against the baseline under matched seeds. The default search space is capped around the lightweight baseline so it tests smaller and moderately larger specifications without returning to the old notebook-scale budgets.

**Question:** can a constrained change to ESCHER hyperparameters produce reliably lower exploitability than the thesis baseline?

### 5. ESCHER warm-start fair ablation

[`experiments/leduc_poker/escher_warm_start_fair_ablation/`](experiments/leduc_poker/escher_warm_start_fair_ablation/README.md)

Runs paired continuous and checkpoint/resume ESCHER arms to test whether interrupting training, saving the full solver state, loading it into a fresh solver, and continuing changes final policy quality. The warm-start boundary defaults to iteration 30, matching the staged exploratory workflow.

**Question:** does checkpoint/resume behaviour introduce an unintended confound relative to an uninterrupted ESCHER baseline with the same headline training budget?

### 6. ESCHER learning-rate schedule ablation

[`experiments/leduc_poker/escher_lr_schedule_ablation/`](experiments/leduc_poker/escher_lr_schedule_ablation/README.md)

Compares the constant-learning-rate ESCHER baseline against a decaying learning-rate schedule under matched seeds and matched training budgets. The default scheduled arm uses cosine decay from the baseline learning rate to 10% of that value.

**Question:** can learning-rate decay stabilise ESCHER's value/regret optimisation enough to reduce exploitability relative to the constant-learning-rate baseline?

### 7. ESCHER average-policy reach-weighting ablation

[`experiments/leduc_poker/escher_reach_weighting_ablation/`](experiments/leduc_poker/escher_reach_weighting_ablation/README.md)

Compares the baseline average-policy regression loss, weighted by CFR iteration only, against a treatment that also weights samples by the acting player's reach probability. Reach multipliers are mean-normalised within each policy-training batch and exclude chance reach.

**Question:** does reach-probability weighting improve the learned average policy produced from ESCHER's average-policy memory?

### 8. ESCHER value-trajectory reuse ablation

[`experiments/leduc_poker/escher_reuse_value_trajectory_ablation/`](experiments/leduc_poker/escher_reuse_value_trajectory_ablation/README.md)

Compares the baseline ESCHER value-data collection scheme, which uses a dedicated history-value traversal pass, against a treatment that reuses player-0 regret traversals to populate the history-value memory. The treatment keeps value-test traversals for diagnostics but removes the dedicated value-training traversal pass.

**Question:** can ESCHER reduce traversal cost by reusing regret trajectories for value training without degrading the learned average policy?

### 9. ESCHER disk-backed regret-memory ablation

[`experiments/leduc_poker/escher_disk_backed_regret_memory_ablation/`](experiments/leduc_poker/escher_disk_backed_regret_memory_ablation/README.md)

Compares the standard in-memory regret replay buffers against a disk-backed TFRecord regret replay backend streamed during regret-network training. Average-policy replay is disk-backed in both arms so the treatment isolates regret-memory storage.

**Question:** can ESCHER reduce regret replay RAM pressure with disk-backed TFRecord shards while preserving strategic performance?

### 10. ESCHER on-policy joint-regret ablation

[`experiments/leduc_poker/escher_on_policy_joint_regret_ablation/`](experiments/leduc_poker/escher_on_policy_joint_regret_ablation/README.md)

Compares the baseline separate player-specific regret traversal batches against an on-policy joint-regret update variant. The treatment samples one trajectory batch from the current joint regret-matching policy and writes regret targets for the acting player at each visited decision node.

**Question:** can ESCHER reduce regret-data generation work by collecting on-policy joint regret samples without degrading the learned average policy?

### 11. ESCHER solver-parameter random search

[`experiments/leduc_poker/escher_solver_parameter_random_search/`](experiments/leduc_poker/escher_solver_parameter_random_search/README.md)

Runs a bounded two-stage random search over configurable ESCHER solver parameters. Screening evaluates the baseline plus sampled solver configurations under a reduced budget; confirmation compares the strongest sampled configurations against the ESCHER baseline with matched seeds and the full baseline budget. The sampled ranges include lightweight and modestly heavier candidates while excluding configurations likely to make multi-seed training time explode.

**Question:** is ESCHER's Leduc poker non-convergence partly caused by a poor balance between traversal budget, value fitting, regret fitting, policy extraction, exploration, and network capacity?

### 12. ESCHER diagnostic hypothesis sweep

[`experiments/leduc_poker/escher_diagnostic_hypothesis_sweep/`](experiments/leduc_poker/escher_diagnostic_hypothesis_sweep/README.md)

Runs a quick single-seed diagnostic sweep over the leading ESCHER exploitability
hypotheses: disabling importance sampling, using uniform zero-regret fallback,
skipping intermediate average-policy extraction, and increasing to a larger
author-style Leduc budget. This experiment is meant for fast insight rather than
thesis-grade multi-seed confirmation.

**Question:** which suspected implementation or parameterisation issue most directly explains ESCHER's poor exploitability convergence?

### 13. ESCHER author-budget multi-seed validation

[`experiments/leduc_poker/escher_author_budget_multiseed/`](experiments/leduc_poker/escher_author_budget_multiseed/README.md)

Runs the best-performing Experiment 12 configuration for 80 ESCHER iterations
over five seeds. This uses the author-style Leduc budget, disables importance
sampling in regret targets, and uses a uniform legal-action zero-regret fallback.

**Question:** does the best single-seed diagnostic configuration retain its
lower exploitability when run for the full baseline iteration budget over a
small multi-seed validation set?

### 14. ESCHER network-size sweep

[`experiments/leduc_poker/escher_network_size_sweep/`](experiments/leduc_poker/escher_network_size_sweep/README.md)

Runs a single-seed sweep over policy, regret, and history-value network
architectures while keeping the Experiment 13 training protocol fixed. The sweep
tests small, lightweight, reference, wider, and deeper multilayer perceptrons.

**Question:** how sensitive is the revised ESCHER configuration to hidden-layer
width and depth?

### 15-22. ESCHER architecture diagnostic sweeps

These single-seed diagnostic experiments extend Experiment 14 by isolating
specific neural-network design choices under the Experiment 13 training budget:

- **Experiment 15:** [`escher_separate_network_architecture_sweep`](experiments/leduc_poker/escher_separate_network_architecture_sweep/README.md) varies relative capacity across the policy, regret, and value networks.
- **Experiment 16:** [`escher_regret_network_width_sweep`](experiments/leduc_poker/escher_regret_network_width_sweep/README.md) varies regret-network width while holding the other networks fixed.
- **Experiment 17:** [`escher_policy_network_width_sweep`](experiments/leduc_poker/escher_policy_network_width_sweep/README.md) varies average-policy-network width while holding the other networks fixed.
- **Experiment 18:** [`escher_layer_norm_ablation`](experiments/leduc_poker/escher_layer_norm_ablation/README.md) tests whether layer normalisation helps or hurts each network.
- **Experiment 19:** [`escher_activation_sweep`](experiments/leduc_poker/escher_activation_sweep/README.md) compares LeakyReLU, ReLU, ELU, GELU, Swish, and Tanh.
- **Experiment 20:** [`escher_residual_mlp_sweep`](experiments/leduc_poker/escher_residual_mlp_sweep/README.md) compares plain MLPs, same-width residual blocks, and projection residual blocks.
- **Experiment 21:** [`escher_bottleneck_architecture_sweep`](experiments/leduc_poker/escher_bottleneck_architecture_sweep/README.md) compares bottleneck, non-bottleneck, wide, and expanding MLP shapes.
- **Experiment 22:** [`escher_shared_trunk_head_sweep`](experiments/leduc_poker/escher_shared_trunk_head_sweep/README.md) compares the current shared trunk plus linear action-output layer with separate per-action heads.

**Question:** which network-design choices most improve ESCHER's exploitability
convergence once the stronger Experiment 13 training protocol is fixed?

### 23. ESCHER regret-target processing ablation

[`experiments/leduc_poker/escher_regret_target_processing_ablation/`](experiments/leduc_poker/escher_regret_target_processing_ablation/README.md)

Tests the Deep CFR Experiment 9 target-processing idea in the ESCHER regret
pathway. The baseline is the carried-forward Experiment 13 protocol plus the
Experiment 22 regret-only action-head architecture. Replay stores raw regret
targets in all variants; standardisation and clipping are applied only to legal
regret-target entries in the supervised regret-network loss.

**Question:** can standardising or clipping sampled ESCHER regret targets reduce
regret-network optimisation variance enough to improve exploitability?

### 24. ESCHER action-head residual-MLP sweep

[`experiments/leduc_poker/escher_action_head_residual_mlp_sweep/`](experiments/leduc_poker/escher_action_head_residual_mlp_sweep/README.md)

Revisits the residual-network hypothesis after the Experiment 22 regret-action
head improvement. The baseline is the carried-forward ESCHER model with
`(256, 128)` trunks and one 64-unit per-action regret head. Treatments add
deeper plain trunks, same-width residual trunks, and projection-residual
bottleneck trunks while keeping the regret action heads fixed.

**Question:** do residual trunk connections still improve ESCHER exploitability
once the regret-output architecture has been strengthened?

### 25. ESCHER average-policy weighting ablation

[`experiments/leduc_poker/escher_average_policy_weighting_ablation/`](experiments/leduc_poker/escher_average_policy_weighting_ablation/README.md)

Tests the Deep CFR Experiment 10 average-strategy weighting idea in ESCHER. The
baseline is the carried-forward ESCHER model with `(256, 128)` trunks, one
64-unit per-action regret head, the Experiment 13 training protocol, no
importance sampling, and uniform zero-regret fallback. The treatment removes
CFR-iteration weighting from the supervised average-policy loss while leaving
all other settings fixed.

**Question:** does uniform average-policy regression weighting improve ESCHER
exploitability relative to the carried-forward linear weighting baseline?

### 26. ESCHER factorised regret-head ablation

[`experiments/leduc_poker/escher_factorised_regret_head_ablation/`](experiments/leduc_poker/escher_factorised_regret_head_ablation/README.md)

Tests the Deep CFR Experiment 16 factorised advantage-head idea in ESCHER. The
baseline is the carried-forward ESCHER model with `(256, 128)` trunks, standard
linear policy output, one 64-unit per-action regret head, the Experiment 13
training protocol, no importance sampling, and uniform zero-regret fallback.
Treatments centre legal-action regret outputs or use a dueling-style scalar
state head plus centred legal-action deviations.

**Question:** does factorising ESCHER's regret-output head improve
exploitability relative to the direct per-action regret-head baseline?

### 27. ESCHER action-head LayerNorm/residual-LN ablation

[`experiments/leduc_poker/escher_action_head_layer_norm_residual_ablation/`](experiments/leduc_poker/escher_action_head_layer_norm_residual_ablation/README.md)

Tests the Deep CFR Experiment 13 layer-normalisation network hypothesis under
the carried-forward ESCHER action-head model. The baseline is the confirmed
`(256, 128)` trunk configuration with standard linear policy output, one
64-unit per-action regret head, the Experiment 13 training protocol, no
importance sampling, and uniform zero-regret fallback. Treatments compare a
same-capacity plain trunk, a deeper plain trunk, a deeper LayerNorm trunk, and
a deeper residual+LayerNorm trunk.

**Question:** does layer normalisation or residual+LayerNorm trunk structure
improve ESCHER exploitability once the regret action-head architecture is fixed?

### 28. ESCHER candidate architecture multi-seed validation

[`experiments/leduc_poker/escher_candidate_architecture_multiseed/`](experiments/leduc_poker/escher_candidate_architecture_multiseed/README.md)

Trains the current best candidate ESCHER architecture over five fixed Leduc
poker seeds. The candidate combines `(256, 256, 128)` plain policy, regret, and
value trunks, no LayerNorm, no residual trunk connections, standard linear
policy output, one 64-unit per-action regret head, and standardised regret
targets without clipping.

**Question:** do the strongest architecture changes identified in the
single-seed diagnostics remain effective when combined and evaluated over five
seeds?

### 29. ESCHER reinitialisation ablation

[`experiments/leduc_poker/escher_reinitialisation_ablation/`](experiments/leduc_poker/escher_reinitialisation_ablation/README.md)

Compares the Experiment 28 candidate architecture against the same model with
regret-network and value-network reinitialisation disabled. Both arms use the
same five fixed development seeds and hold the architecture, target processing,
traversal budget, replay settings, and supervised update budgets fixed.

**Question:** does keeping the regret and value networks persistent across
ESCHER iterations improve the Experiment 28 candidate architecture?

### 30. ESCHER candidate learning-rate ablation

[`experiments/leduc_poker/escher_learning_rate_ablation/`](experiments/leduc_poker/escher_learning_rate_ablation/README.md)

Compares the Experiment 28 candidate architecture against low and high constant
learning-rate treatments. The baseline uses the carried-forward constant
learning rate \(10^{-3}\), while the treatments use \(5\times10^{-4}\) and
\(2\times10^{-3}\). No learning-rate decay schedules are tested.

**Question:** does a lower or higher constant learning rate improve the
Experiment 28 candidate architecture?

### 31. ESCHER candidate replay-capacity ablation

[`experiments/leduc_poker/escher_replay_capacity_ablation/`](experiments/leduc_poker/escher_replay_capacity_ablation/README.md)

Compares the Experiment 28 candidate architecture against larger replay-buffer
treatments. The baseline uses replay capacity 50,000 with regret and value
batches of 256. Treatments test replay capacity 100,000, replay capacity
200,000, and replay capacity 200,000 with regret batch size increased to 512
while the value batch remains 256.

**Question:** does relieving replay-buffer pressure, with or without a larger
regret supervised batch, allow the Experiment 28 candidate architecture to keep
improving later in training?

### 32. ESCHER candidate regret/value work-balance ablation

[`experiments/leduc_poker/escher_regret_value_work_ablation/`](experiments/leduc_poker/escher_regret_value_work_ablation/README.md)

Compares the Experiment 28 candidate architecture against regret-heavy work
allocations. The baseline uses 500 regret traversals, 500 value traversals, 200
regret-network training steps, and 200 value-network training steps. Treatments
shift traversal collection, supervised fitting effort, or both toward the regret
pathway while keeping the architecture fixed.

**Question:** is the Experiment 28 plateau partly caused by insufficient regret
data or regret fitting relative to history-value fitting?

### 33. ESCHER candidate regret action-head capacity ablation

[`experiments/leduc_poker/escher_regret_action_head_capacity_ablation/`](experiments/leduc_poker/escher_regret_action_head_capacity_ablation/README.md)

Compares the Experiment 28 candidate regret head against two larger per-action
regret-head designs: one 128-unit head and a two-layer 64-by-64 head. The policy
network, value network, trunk architecture, replay settings, and training
protocol remain fixed.

**Question:** does increasing the capacity of the per-action regret head improve
ESCHER's ability to model regret targets after the shared trunk?

### 34. ESCHER candidate regret-batch-size ablation

[`experiments/leduc_poker/escher_regret_batch_size_ablation/`](experiments/leduc_poker/escher_regret_batch_size_ablation/README.md)

Compares the Experiment 28 candidate architecture against the same model with
regret-network batch size increased from 256 to 512. Replay capacity remains at
50,000 and the value-network batch remains 256, isolating the batch-size signal
that was confounded with larger replay capacity in Experiment 31.

**Question:** does increasing only the regret supervised batch size improve the
Experiment 28 candidate architecture at the baseline replay capacity?

Quick local smoke test:

```bash
python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run \
  --seeds 1234 \
  --variant-ids baseline_regret_batch_256,regret_batch_512 \
  --iterations 2 \
  --traversals 2 \
  --value-traversals 2 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

### 35. ESCHER candidate regret-target specification ablation

[`experiments/leduc_poker/escher_regret_target_specification_ablation/`](experiments/leduc_poker/escher_regret_target_specification_ablation/README.md)

Compares the exact Experiment 28 author-code target
`Q_hat(h,a) - V_hat(h)` against the policy-weighted child-Q target specified in
Equation 7 / Algorithm 2 of the ESCHER paper. Both arms retain Experiment 28's
architecture, seeds, budgets, replay settings, and standardized target
processing. Bellman-consistency and raw-target centering diagnostics are
exported alongside exploitability.

**Question:** does enforcing a policy-consistent raw regret target remove part
of the Experiment 28 exploitability plateau?

### 36. ESCHER corrected regret-target scale-only ablation

[`experiments/leduc_poker/escher_regret_target_scale_only_ablation/`](experiments/leduc_poker/escher_regret_target_scale_only_ablation/README.md)

Uses the exact Experiment 28 configuration as the primary baseline and tests
corrected policy-weighted regret targets under raw, fixed utility-range,
minibatch-RMS, persistent-standard-deviation, and legacy batch-centred
processing. It exports target sign-flip and positive-target diagnostics, plus
paired comparisons against both Experiment 28 and the corrected batch-centred
control.

**Question:** can scale-only conditioning retain Experiment 28's optimization
benefit without changing target signs and positive-regret geometry?

### 37. ESCHER 2x2 regret-target correction factorial

[`experiments/leduc_poker/escher_regret_target_factorial_correction/`](experiments/leduc_poker/escher_regret_target_factorial_correction/README.md)

Runs the exact Experiment 28 baseline, policy-weighted-Q correction only,
scale-only minibatch-RMS normalization only, and both corrections. All four
arms are screened on three seeds. The two best treatments and the baseline are
then confirmed on five separate seeds. The primary success criterion is
exploitability below `0.3` at approximately one million nodes.

**Question:** which correction drives improvement, and do the two corrections
interact constructively?

### 38. ESCHER regret replay composition ablation

[`experiments/leduc_poker/escher_regret_replay_composition_ablation/`](experiments/leduc_poker/escher_regret_replay_composition_ablation/README.md)

Compares Experiment 28's uniform regret reservoir with append-only all-sample
replay, infoset-stratified replay, protected rare-infoset quotas, and
counterfactual-reach-weighted priority replay. Regret optimizer work, value
replay, and average-policy replay remain fixed, while replay saturation and
infoset-balance diagnostics are exported by nodes touched.

**Question:** does removing or restructuring finite regret replay move the
Experiment 28 exploitability plateau?

### 39. ESCHER fixed sampling-policy coverage ablation

[`experiments/leduc_poker/escher_fixed_sampling_coverage_ablation/`](experiments/leduc_poker/escher_fixed_sampling_coverage_ablation/README.md)

Compares Experiment 28's fixed uniform-action sampling with exact
leaf-balanced sampling and a fixed 50/50 tempered mixture. All algorithmic
settings and optimizer work remain fixed, while exact and empirical history
coverage diagnostics are exported by nodes touched.

**Question:** does improving the minimum fixed-policy reach of Leduc histories
materially lower the Experiment 28 exploitability plateau?

### 40. ESCHER sequential/parallel equivalence ablation

[`experiments/leduc_poker/escher_parallel_equivalence_ablation/`](experiments/leduc_poker/escher_parallel_equivalence_ablation/README.md)

Compares the exact Experiment 28 sequential solver with a three-worker
Ray-parallel experience-collection backend adapted from Sandholm-Lab's public
implementation. Total traversals, replay capacity, learner updates, and three
paired seeds are held fixed. It reports pre-declared final-policy equivalence
tests plus experience-collection-phase and end-to-end speedups.

**Question:** does parallel collection preserve solution quality while reducing
runtime, without giving the parallel arm a larger data or replay budget?

### 41. ESCHER combined candidate ablation

[`experiments/leduc_poker/escher_combined_candidate_ablation/`](experiments/leduc_poker/escher_combined_candidate_ablation/README.md)

Compares the exact Experiment 28 baseline with policy-weighted-Q targets plus
infoset-stratified replay, first under uniform fixed sampling and then with
exact balanced fixed sampling. All other Experiment 28 settings remain fixed,
and the three arms use three matched seeds.

**Question:** do the strongest target and replay treatments combine, and does
exact balanced sampling add value once both are enabled?

### 42. ESCHER 10x-node long-horizon candidate ablation

[`experiments/leduc_poker/escher_long_horizon_candidate_ablation/`](experiments/leduc_poker/escher_long_horizon_candidate_ablation/README.md)

Compares the exact Experiment 28 algorithm with the policy-weighted-Q,
infoset-stratified, uniform-sampling candidate over three matched seeds. Both
arms execute 801 solve passes, approximately 9.89 times Experiment 28's
solve-pass, traversal, and per-pass optimizer-event budget; the maximum stack
is excluded.

The long-horizon runner uses memory-stable in-place network and optimizer
reinitialization, while the GCP submitter provides transient-VM retries,
periodic output uploads, and resource heartbeats for multi-day runs.

**Question:** does substantially longer training lower either plateau, and does
the combined candidate develop a durable advantage near 9.3 million nodes?

## Setup

Create and activate a Python 3.9 virtual environment. The repository contains
a placeholder `venv/` directory, but the actual environment is not committed.

```bash
python3.9 -m venv venv
source venv/bin/activate       # macOS/Linux
# .\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

OpenSpiel installation can vary by platform. If `pip install -r requirements.txt` fails on `open_spiel`, install OpenSpiel following the official instructions for your platform.

## Running the experiments

From the repository root:

```bash
# Experiment 1 — full aligned ESCHER baseline
python -m experiments.leduc_poker.escher_multiseed_baseline.run

# Experiment 2 — intermediate policy-training ablation
python -m experiments.leduc_poker.escher_intermediate_policy_training_ablation.run

# Experiment 3 — checkpoint-stability head-to-head analysis
python -m experiments.leduc_poker.escher_checkpoint_stability.run

# Experiment 4 — constrained hyperparameter search
python -m experiments.leduc_poker.escher_constrained_hyperparameter_search.run

# Experiment 5 — warm-start/checkpoint-resume fair ablation
python -m experiments.leduc_poker.escher_warm_start_fair_ablation.run

# Experiment 6 — learning-rate schedule ablation
python -m experiments.leduc_poker.escher_lr_schedule_ablation.run

# Experiment 7 — average-policy reach-weighting ablation
python -m experiments.leduc_poker.escher_reach_weighting_ablation.run

# Experiment 8 — value-trajectory reuse ablation
python -m experiments.leduc_poker.escher_reuse_value_trajectory_ablation.run

# Experiment 9 — disk-backed regret-memory ablation
python -m experiments.leduc_poker.escher_disk_backed_regret_memory_ablation.run

# Experiment 10 — on-policy joint-regret ablation
python -m experiments.leduc_poker.escher_on_policy_joint_regret_ablation.run

# Experiment 11 — solver-parameter random search
python -m experiments.leduc_poker.escher_solver_parameter_random_search.run

# Experiment 12 — quick diagnostic hypothesis sweep
python -m experiments.leduc_poker.escher_diagnostic_hypothesis_sweep.run

# Experiment 13 — author-budget multi-seed validation
python -m experiments.leduc_poker.escher_author_budget_multiseed.run

# Experiment 14 — network-size sweep
python -m experiments.leduc_poker.escher_network_size_sweep.run

# Experiment 15 — separate network architecture sweep
python -m experiments.leduc_poker.escher_separate_network_architecture_sweep.run

# Experiment 16 — regret-network width sweep
python -m experiments.leduc_poker.escher_regret_network_width_sweep.run

# Experiment 17 — policy-network width sweep
python -m experiments.leduc_poker.escher_policy_network_width_sweep.run

# Experiment 18 — layer-normalisation ablation
python -m experiments.leduc_poker.escher_layer_norm_ablation.run

# Experiment 19 — activation-function sweep
python -m experiments.leduc_poker.escher_activation_sweep.run

# Experiment 20 — residual-MLP sweep
python -m experiments.leduc_poker.escher_residual_mlp_sweep.run

# Experiment 21 — bottleneck architecture sweep
python -m experiments.leduc_poker.escher_bottleneck_architecture_sweep.run

# Experiment 22 — shared-trunk/action-head sweep
python -m experiments.leduc_poker.escher_shared_trunk_head_sweep.run

# Experiment 23 — regret-target processing ablation
python -m experiments.leduc_poker.escher_regret_target_processing_ablation.run

# Experiment 24 — action-head residual-MLP sweep
python -m experiments.leduc_poker.escher_action_head_residual_mlp_sweep.run

# Experiment 25 — average-policy weighting ablation
python -m experiments.leduc_poker.escher_average_policy_weighting_ablation.run

# Experiment 26 — factorised regret-head ablation
python -m experiments.leduc_poker.escher_factorised_regret_head_ablation.run

# Experiment 27 — action-head LayerNorm/residual-LN ablation
python -m experiments.leduc_poker.escher_action_head_layer_norm_residual_ablation.run

# Experiment 28 — candidate architecture multi-seed validation
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run

# Experiment 29 — reinitialisation ablation on the candidate architecture
python -m experiments.leduc_poker.escher_reinitialisation_ablation.run

# Experiment 30 — learning-rate ablation on the candidate architecture
python -m experiments.leduc_poker.escher_learning_rate_ablation.run

# Experiment 31 — replay-capacity ablation on the candidate architecture
python -m experiments.leduc_poker.escher_replay_capacity_ablation.run

# Experiment 32 — regret/value work-balance ablation on the candidate architecture
python -m experiments.leduc_poker.escher_regret_value_work_ablation.run

# Experiment 33 — regret action-head capacity ablation on the candidate architecture
python -m experiments.leduc_poker.escher_regret_action_head_capacity_ablation.run

# Experiment 34 — regret-batch-size ablation on the candidate architecture
python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run

# Experiment 35 — regret-target specification ablation on the candidate architecture
python -m experiments.leduc_poker.escher_regret_target_specification_ablation.run

# Experiment 36 — corrected regret-target scale-only ablation
python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run

# Experiment 37 — staged 2x2 regret-target correction factorial
python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run

# Experiment 38 — regret replay composition ablation
python -m experiments.leduc_poker.escher_regret_replay_composition_ablation.run

# Experiment 39 — fixed sampling-policy coverage ablation
python -m experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run

# Experiment 40 — sequential versus Ray-parallel equivalence ablation
python -m experiments.leduc_poker.escher_parallel_equivalence_ablation.run

# Experiment 41 — combined candidate versus Experiment 28
python -m experiments.leduc_poker.escher_combined_candidate_ablation.run

# Experiment 42 — 10x-node long-horizon candidate comparison
python -m experiments.leduc_poker.escher_long_horizon_candidate_ablation.run
```

For quick GCP smoke tests, first make sure the environment variables required
by `gcp/submit_batch_experiment.sh` are set: `PROJECT_ID`, `REGION`, `BUCKET`,
and `SA_EMAIL`. Then paste any of the following commands from the repository
root. Each command submits a separate Google Batch job.

Batch tasks retry transient VM failures (`50001`-`50004`) twice and upload live
outputs every five minutes by default. Set `BATCH_DRY_RUN=true` to render and
inspect a job without submitting it; `BATCH_MAX_RETRY_COUNT` and
`BATCH_OUTPUT_UPLOAD_INTERVAL_SECONDS` override the reliability defaults.

```bash
# Experiment 1 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp1-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_multiseed_baseline.run \
    --seeds 1234,2025 \
    --iterations 10 \
    --traversals 50 \
    --value-traversals 50 \
    --policy-network-train-steps 20 \
    --regret-network-train-steps 20 \
    --value-network-train-steps 20 \
    --evaluation-interval 5 \
    --output-root outputs/cloud/escher-smoke-exp1" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 2 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp2-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_intermediate_policy_training_ablation.run \
    --seeds 1234 \
    --iterations 10 \
    --traversals 50 \
    --value-traversals 50 \
    --policy-network-train-steps 20 \
    --regret-network-train-steps 20 \
    --value-network-train-steps 20 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp2" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 3 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp3-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_checkpoint_stability.run \
    --seeds 1234 \
    --checkpoint-schedule 1,2 \
    --traversals 50 \
    --value-traversals 50 \
    --policy-network-train-steps 20 \
    --regret-network-train-steps 20 \
    --value-network-train-steps 20 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp3" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 4 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp4-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_constrained_hyperparameter_search.run \
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
    --output-root outputs/cloud/escher-smoke-exp4" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 5 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp5-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_warm_start_fair_ablation.run \
    --seeds 1234 \
    --iterations 2 \
    --warm-start-boundary 1 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp5" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 6 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp6-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_lr_schedule_ablation.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp6" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 7 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp7-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_reach_weighting_ablation.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp7" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 8 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp8-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_reuse_value_trajectory_ablation.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 5 \
    --value-traversals 5 \
    --value-test-traversals 2 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp8" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 9 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp9-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_disk_backed_regret_memory_ablation.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp9" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 10 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp10-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_on_policy_joint_regret_ablation.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp10" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 11 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp11-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_solver_parameter_random_search.run \
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
    --output-root outputs/cloud/escher-smoke-exp11" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 23 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp23-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_processing_ablation.run \
    --seeds 1234 \
    --variant-ids raw_regret_targets,standardized_clipped_regret_targets \
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
    --memory-capacity 128 \
    --output-root outputs/cloud/escher-smoke-exp23" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 24 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp24-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_action_head_residual_mlp_sweep.run \
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
    --output-root outputs/cloud/escher-smoke-exp24" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 25 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp25-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_average_policy_weighting_ablation.run \
    --seeds 1234 \
    --variant-ids linear_avg_weighting_baseline,uniform_avg_weighting \
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
    --memory-capacity 128 \
    --output-root outputs/cloud/escher-smoke-exp25" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 26 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp26-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_factorised_regret_head_ablation.run \
    --seed 1234 \
    --variant-ids direct_regret_action_head_64_baseline,dueling_regret_action_head_64 \
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
    --output-root outputs/cloud/escher-smoke-exp26" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 27 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp27-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_action_head_layer_norm_residual_ablation.run \
    --seed 1234 \
    --variant-ids carry_forward_layer_norm_256_128_action_heads,deep_residual_layer_norm_256_256_128_action_heads \
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
    --output-root outputs/cloud/escher-smoke-exp27" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 28 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp28-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run \
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
    --output-root outputs/cloud/escher-smoke-exp28" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 29 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp29-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_reinitialisation_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp29" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 30 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp30-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_learning_rate_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp30" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 31 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp31-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_replay_capacity_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_replay_50k \
    --iterations 2 \
    --traversals 2 \
    --value-traversals 2 \
    --policy-network-train-steps 1 \
    --regret-network-train-steps 1 \
    --value-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp31" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 32 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp32-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_value_work_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_regret_value_work \
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
    --output-root outputs/cloud/escher-smoke-exp32" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 33 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp33-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_action_head_capacity_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_regret_head_64 \
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
    --output-root outputs/cloud/escher-smoke-exp33" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 34 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp34-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_batch_size_ablation.run \
    --seeds 1234 \
    --variant-ids baseline_regret_batch_256 \
    --iterations 2 \
    --traversals 2 \
    --value-traversals 2 \
    --policy-network-train-steps 1 \
    --regret-network-train-steps 1 \
    --value-network-train-steps 1 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp34" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 35 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp35-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_specification_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp35" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 36 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp36-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_scale_only_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp36" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 37 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp37-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_target_factorial_correction.run \
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
    --output-root outputs/cloud/escher-smoke-exp37" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 38 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp38-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_regret_replay_composition_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp38" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 39 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp39-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_fixed_sampling_coverage_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp39" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 40 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp40-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_parallel_equivalence_ablation.run \
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
    --output-root outputs/cloud/escher-smoke-exp40" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 41 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp41-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_combined_candidate_ablation.run \
    --seeds 1234 \
    --variant-ids experiment_28_baseline,policy_q_stratified_uniform,policy_q_stratified_exact_balanced \
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
    --output-root outputs/cloud/escher-smoke-exp41" \
  "n2-standard-4" "3600" "4000" "16000" "100"

# Experiment 42 smoke test
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp42-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_long_horizon_candidate_ablation.run \
    --seeds 1234 \
    --variant-ids experiment_28_10x_nodes,policy_q_stratified_uniform_10x_nodes \
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
    --output-root outputs/cloud/escher-smoke-exp42" \
  "n2-standard-4" "3600" "4000" "16000" "100"
```

Outputs are written to a timestamped subdirectory under `outputs/` by default. The key files are:

Average-policy-value charts plot the configured `average_policy_value_target`.
For Leduc poker this is player 0's Nash equilibrium value, approximately
`-0.085606424078`; future games should override that config value rather than
editing plotting code.

```text
seed_summary.csv
aggregate_summary.json
checkpoint_curves.csv
experiment_metadata.json
exploitability_by_iteration_multiseed.png
exploitability_by_nodes_multiseed.png
average_policy_value_by_iteration_multiseed.png
average_policy_value_by_nodes_multiseed.png
policy_value_error_multiseed.png
policy_loss_diagnostic.png
regret_loss_diagnostic.png
value_loss_diagnostic.png
```

Full `outputs/` directories are scratch working data and are gitignored. Curated
lightweight thesis-facing results can be promoted into the tracked
`thesis_artifacts/` tree with [docs/THESIS_ARTIFACTS.md](docs/THESIS_ARTIFACTS.md).

Ablation experiments also export variant-level and paired-comparison files such as:

```text
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
paired_difference_summary.json
final_exploitability_by_variant.png
final_average_policy_value_by_variant.png
runtime_by_variant.png
```

Checkpoint-stability experiments also export policy snapshots, exact pairwise head-to-head matrices, monotonicity summaries, and checkpoint-strength plots such as:

```text
checkpoint_stage_summary.csv
checkpoint_exploitability_metrics.csv
checkpoint_average_policy_value_with_target.png
head_to_head_exact_pairwise.csv
head_to_head_exact_mean_matrix.csv
head_to_head_monotonicity_summary_by_seed.csv
head_to_head_strength_vs_earlier_aggregate.png
head_to_head_later_vs_earlier_matrix.png
```

Hyperparameter-search experiments export screening and confirmation summaries, paired confirmation deltas, and stage-level plots such as:

```text
screening_seed_summary.csv
screening_aggregate_by_variant.csv
confirmation_seed_summary.csv
confirmation_aggregate_by_variant.csv
confirmation_paired_differences_vs_baseline.csv
confirmation_paired_difference_summary.csv
screening_exploitability_by_iteration.png
screening_average_policy_value_by_iteration.png
confirmation_final_exploitability_by_variant.png
confirmation_final_average_policy_value_by_variant.png
```

Warm-start ablations export paired continuous/resumed summaries and checkpoint-resume artifacts such as:

```text
seed_summary.csv
paired_summary.csv
paired_aggregate_summary.csv
paired_checkpoint_differences.csv
warm_start_exploitability_by_iteration.png
warm_start_average_policy_value_by_iteration.png
warm_start_paired_delta_exploitability_warm_minus_baseline.png
checkpoints/
```

Learning-rate schedule ablations export schedule-level summaries, paired deltas, active learning-rate curves, and diagnostics such as:

```text
seed_summary.csv
schedule_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
lr_schedule_learning_rates.png
lr_schedule_exploitability_by_iteration.png
lr_schedule_average_policy_value_by_iteration.png
lr_schedule_paired_delta_final_exploitability.png
```

Reach-weighting ablations export matched variant summaries, reach diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_reach_ablation.png
average_policy_value_by_iteration_reach_ablation.png
final_average_policy_value_reach_ablation.png
paired_final_exploitability_delta_reach_minus_baseline.png
```

Value-trajectory reuse ablations export matched variant summaries, traversal-budget diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_reuse_ablation.png
average_policy_value_by_iteration_reuse_ablation.png
final_average_policy_value_reuse_ablation.png
dedicated_value_traversals_reuse_ablation.png
paired_final_exploitability_delta_reuse_minus_baseline.png
```

Disk-backed regret-memory ablations export matched variant summaries, memory/storage diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_regret_memory_ablation.png
average_policy_value_by_iteration_regret_memory_ablation.png
final_average_policy_value_regret_memory_ablation.png
peak_rss_by_variant.png
regret_storage_mb_by_variant.png
paired_final_exploitability_delta_disk_minus_baseline.png
replay/
```

On-policy joint-regret ablations export matched variant summaries, traversal-budget diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_on_policy_ablation.png
average_policy_value_by_iteration_on_policy_ablation.png
final_average_policy_value_on_policy_ablation.png
nominal_regret_traversals_by_variant.png
paired_final_exploitability_delta_on_policy_minus_baseline.png
```

Solver-parameter random searches export screening and confirmation summaries, paired confirmation deltas, stage-specific curves, sampled-parameter metadata, and plots such as:

```text
screening_seed_summary.csv
screening_aggregate_by_variant.csv
confirmation_seed_summary.csv
confirmation_aggregate_by_variant.csv
confirmation_paired_differences_vs_baseline.csv
screening_checkpoint_curves.csv
confirmation_checkpoint_curves.csv
solver_parameter_random_search_curves.npz
screening_exploitability_by_iteration.png
screening_average_policy_value_by_iteration.png
confirmation_paired_delta_final_exploitability.png
```

## Notes for adding future experiments

When adding a new ESCHER experiment, follow the same pattern as the baseline:

1. create a new folder under `experiments/leduc_poker/`;
2. include a `config.py`, `run.py`, and `README.md`;
3. hold the baseline protocol fixed except for the intended treatment variable;
4. use matched seeds where possible;
5. export the same core metrics and plots so the thesis results have consistent look and feel.
6. put reusable seed runners, metrics, and plotting helpers in `escher_poker` when they will likely apply to more than one experiment.
