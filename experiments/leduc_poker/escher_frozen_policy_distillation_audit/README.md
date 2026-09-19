# Experiment 45: Frozen-Reservoir Distillation Audit

## Purpose

This experiment tests whether standard ESCHER's apparent Leduc plateau is
substantially caused by average-policy approximation rather than by the
underlying regret learner.

Three development seeds (`1234`, `2025`, and `31415`) use the selected standard
ESCHER architecture and the Experiment 44 training budget: 1,300 configured
iterations, approximately 15 million touched nodes and approximately 12 hours
per seed on `n2-standard-8`. The ten-iteration evaluation and policy-fitting
cadence is also retained. The regret and transient value memories remain at
50,000 rows. Only the lifetime average-policy reservoir is enlarged to
1,000,000 rows.

After each source run, the final reservoir is frozen to a compressed NPZ file,
checksummed, reloaded, grouped by information set, and evaluated exactly. Four
fresh `256x256x128` policy networks are then fitted from the identical frozen
source and identical initialization:

1. row-wise iteration-weighted MSE at the existing fitting budget;
2. row-wise iteration-weighted soft-target cross-entropy with the same number
   of network-example presentations;
3. grouped iteration-weighted soft-target cross-entropy with matched
   network-example presentations;
4. the grouped objective with four times as many presentations.

For grouped fitting, one row is the sufficient statistic for an observed
information set. Its objective weight makes grouped soft-target
cross-entropy algebraically identical to row-wise soft-target
cross-entropy. The matched-example arms control the number of actual network
input rows processed, not merely the number of optimiser steps.

## Local smoke test

From the repository root:

```bash
python -m experiments.leduc_poker.escher_frozen_policy_distillation_audit.run \
  --smoke \
  --output-root outputs/smoke_tests
```

The smoke test runs one tiny seed, freezes and reloads its reservoir, fits all
four arms, performs exact Leduc evaluations, and writes both charts.

## Full local run

```bash
python -m experiments.leduc_poker.escher_frozen_policy_distillation_audit.run
```

The three seeds run sequentially. Expect approximately 39--48 elapsed hours on
hardware comparable to `n2-standard-8`, including offline fitting. Results are
rewritten after every completed seed so a later failure does not erase earlier
evidence.

## GCP Batch

With the same `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` environment
variables used by the earlier ESCHER experiments:

```bash
./gcp/submit_batch_experiment.sh \
  "leduc-escher-exp45-distill-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.leduc_poker.escher_frozen_policy_distillation_audit.run \
    --output-root outputs/cloud/leduc-escher-exp45-distill" \
  "n2-standard-8" "216000" "8000" "32000" "150"
```

The 60-hour hard limit bounds cost while leaving margin above the expected
three-seed sequential runtime.

## Principal outputs

| Output | Contents |
| --- | --- |
| `seed_<seed>/frozen_average_policy_reservoir.npz` | Lossless frozen final reservoir used by every arm. |
| `source_seed_metrics.csv` | Source training, reservoir, empirical-policy and archived neural-policy diagnostics. |
| `fit_metrics.csv` | Exact exploitability, empirical gap, fitting work and runtime for every arm and seed. |
| `arm_summary.csv` | Cross-seed means, standard deviations and standard errors. |
| `exploitability_by_distillation_arm.png` | Exact arm comparison with the empirical-reservoir reference. |
| `distillation_gap_by_arm.png` | Neural minus empirical-reservoir exploitability. |
| `aggregate_summary.json` | Machine-readable experiment summary. |
