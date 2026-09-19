# Experiment 46: ESCHER Paper-Hyperparameter Distillation Audit

## Purpose

Experiment 46 is a controlled derivative of Experiment 45. It asks whether the
substantially larger optimisation budget reported for the principal deep
experiments in the ESCHER paper improves standard ESCHER on Leduc. It changes
only the parameters explicitly reported in Appendix 10.1, Table 3:

| Parameter | Experiment 45 | Experiment 46 |
| --- | ---: | ---: |
| Regret traversals | 500 | 1,000 |
| History-value traversals | 500 | 1,000 |
| Regret minibatch | 256 | 2,048 |
| History-value minibatch | 256 | 2,048 |
| Regret-network steps | 200 | 5,000 |
| History-value-network steps | 200 | 5,000 |
| Average-policy steps | 1,000 | 10,000 |

Everything else is inherited: seeds `1234`, `2025`, and `31415`; the
`256x256x128` networks; the 50,000-row regret/value memories; the
1,000,000-row lifetime average-policy reservoir; the four frozen-reservoir
distillation arms; and the 12-hour active-training endpoint. Consequently,
iterations and touched nodes are measured outcomes rather than fixed budgets.
An in-flight iteration is completed after the boundary.

The paper settings were selected for its deep Phantom Tic-Tac-Toe and Dark Hex
experiments, while its reported Leduc evaluation was tabular with an oracle
value function. Experiment 46 should therefore be described as applying the
paper's deep-training hyperparameters to Leduc, not as an exact paper
replication.

## Mandatory local smoke test

From the repository root:

```bash
./gcp/run_escher_paper_hyperparameter_distillation_audit.sh smoke-local
```

Smoke mode replaces every expensive dimension with a tiny value while testing
the Experiment 46 contract, worker manifest, frozen-reservoir checksum,
four-arm fitting path and aggregation.

## Full local run

```bash
python -m experiments.leduc_poker.escher_paper_hyperparameter_distillation_audit.run
```

Local seeds run sequentially and are not recommended for the production study.

## GCP Batch

After the smoke-tested commit has been pushed, reuse the established cloud
configuration:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp46-paper-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_escher_paper_hyperparameter_distillation_audit.sh run
```

The cloud-owned controller runs smoke, then the three seeds concurrently on
separate `n2-standard-8` VMs, then aggregation. The laptop may be disconnected
after submission. Monitor or resume with:

```bash
./gcp/run_escher_paper_hyperparameter_distillation_audit.sh status
./gcp/run_escher_paper_hyperparameter_distillation_audit.sh resume
```

Each worker has a 48-hour hard ceiling. This is a cost bound rather than a
runtime prediction: the source trajectory has a 12-hour budget, but final
policy fitting and the four frozen-reservoir arms are substantially heavier
than in Experiment 45.

## Principal outputs

The output schema is identical to Experiment 45. In particular,
`source_seed_metrics.csv` records the realised time, endpoint overshoot,
completed iterations and touched nodes, while `fit_metrics.csv` and the two
charts report exact exploitability and the empirical-reservoir distillation
gap for every arm.

