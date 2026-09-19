#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NUMBER=48
export EXPERIMENT_RUN_PREFIX="exp48-all1m"
export EXPERIMENT_DRY_RUN_PREFIX="exp48"
export EXPERIMENT_REMOTE_CONTROLLER_VAR="EXP48_REMOTE_CONTROLLER"
export EXPERIMENT_SMOKE_MODULE="experiments.leduc_poker.escher_all_large_memory_distillation_audit.cloud"
export EXPERIMENT_BATCH_BUILDER="$SCRIPT_DIR/escher_all_large_memory_distillation_audit_batch.py"

exec bash "$SCRIPT_DIR/run_escher_frozen_policy_distillation_audit.sh" "$@"
