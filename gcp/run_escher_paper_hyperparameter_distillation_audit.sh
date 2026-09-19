#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NUMBER=46
export EXPERIMENT_RUN_PREFIX="exp46-paper"
export EXPERIMENT_DRY_RUN_PREFIX="exp46"
export EXPERIMENT_REMOTE_CONTROLLER_VAR="EXP46_REMOTE_CONTROLLER"
export EXPERIMENT_SMOKE_MODULE="experiments.leduc_poker.escher_paper_hyperparameter_distillation_audit.cloud"
export EXPERIMENT_BATCH_BUILDER="$SCRIPT_DIR/escher_paper_hyperparameter_distillation_audit_batch.py"

exec bash "$SCRIPT_DIR/run_escher_frozen_policy_distillation_audit.sh" "$@"

