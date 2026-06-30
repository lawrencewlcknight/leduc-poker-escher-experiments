#!/usr/bin/env bash
set -euo pipefail

# Runs compact smoke tests for the five most recent ESCHER experiments:
# 23. regret-target processing
# 24. action-head residual MLP
# 25. average-policy weighting
# 26. factorised regret head
# 27. action-head LayerNorm/residual-LN
#
# Intended for use inside a Google Batch VM via:
#   bash gcp/run_recent_experiment_smoke_tests.sh
#
# Environment overrides:
#   SMOKE_OUTPUT_ROOT, SMOKE_SEED, SMOKE_ITERATIONS, SMOKE_TRAVERSALS,
#   SMOKE_VALUE_TRAVERSALS, SMOKE_EVALUATION_INTERVAL,
#   SMOKE_POLICY_TRAIN_STEPS, SMOKE_REGRET_TRAIN_STEPS,
#   SMOKE_VALUE_TRAIN_STEPS, SMOKE_BATCH_SIZE, SMOKE_MEMORY_CAPACITY

SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-outputs/cloud/escher-recent-smoke}"
SMOKE_SEED="${SMOKE_SEED:-1234}"
SMOKE_ITERATIONS="${SMOKE_ITERATIONS:-2}"
SMOKE_TRAVERSALS="${SMOKE_TRAVERSALS:-2}"
SMOKE_VALUE_TRAVERSALS="${SMOKE_VALUE_TRAVERSALS:-2}"
SMOKE_EVALUATION_INTERVAL="${SMOKE_EVALUATION_INTERVAL:-1}"
SMOKE_POLICY_TRAIN_STEPS="${SMOKE_POLICY_TRAIN_STEPS:-1}"
SMOKE_REGRET_TRAIN_STEPS="${SMOKE_REGRET_TRAIN_STEPS:-1}"
SMOKE_VALUE_TRAIN_STEPS="${SMOKE_VALUE_TRAIN_STEPS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_MEMORY_CAPACITY="${SMOKE_MEMORY_CAPACITY:-128}"

run_module() {
  local title="$1"
  local module="$2"
  shift 2

  echo
  echo "================================================================"
  echo "Running ${title}"
  echo "Module: ${module}"
  echo "================================================================"
  python -m "${module}" "$@"
}

COMMON_ARGS=(
  --iterations "${SMOKE_ITERATIONS}"
  --traversals "${SMOKE_TRAVERSALS}"
  --value-traversals "${SMOKE_VALUE_TRAVERSALS}"
  --evaluation-interval "${SMOKE_EVALUATION_INTERVAL}"
  --policy-network-train-steps "${SMOKE_POLICY_TRAIN_STEPS}"
  --regret-network-train-steps "${SMOKE_REGRET_TRAIN_STEPS}"
  --value-network-train-steps "${SMOKE_VALUE_TRAIN_STEPS}"
  --batch-size-regret "${SMOKE_BATCH_SIZE}"
  --batch-size-value "${SMOKE_BATCH_SIZE}"
  --batch-size-average-policy "${SMOKE_BATCH_SIZE}"
)

run_module \
  "Experiment 23: regret-target processing" \
  "experiments.leduc_poker.escher_regret_target_processing_ablation.run" \
  --seeds "${SMOKE_SEED}" \
  --variant-ids raw_regret_targets,standardized_clipped_regret_targets \
  "${COMMON_ARGS[@]}" \
  --memory-capacity "${SMOKE_MEMORY_CAPACITY}" \
  --output-root "${SMOKE_OUTPUT_ROOT}/exp23_regret_target_processing"

run_module \
  "Experiment 24: action-head residual MLP" \
  "experiments.leduc_poker.escher_action_head_residual_mlp_sweep.run" \
  --seed "${SMOKE_SEED}" \
  --variant-ids carry_forward_256_128_action_heads,deep_same_width_256_256_128_action_heads \
  "${COMMON_ARGS[@]}" \
  --output-root "${SMOKE_OUTPUT_ROOT}/exp24_action_head_residual_mlp"

run_module \
  "Experiment 25: average-policy weighting" \
  "experiments.leduc_poker.escher_average_policy_weighting_ablation.run" \
  --seeds "${SMOKE_SEED}" \
  --variant-ids linear_avg_weighting_baseline,uniform_avg_weighting \
  "${COMMON_ARGS[@]}" \
  --memory-capacity "${SMOKE_MEMORY_CAPACITY}" \
  --output-root "${SMOKE_OUTPUT_ROOT}/exp25_average_policy_weighting"

run_module \
  "Experiment 26: factorised regret head" \
  "experiments.leduc_poker.escher_factorised_regret_head_ablation.run" \
  --seed "${SMOKE_SEED}" \
  --variant-ids direct_regret_action_head_64_baseline,dueling_regret_action_head_64 \
  "${COMMON_ARGS[@]}" \
  --output-root "${SMOKE_OUTPUT_ROOT}/exp26_factorised_regret_head"

run_module \
  "Experiment 27: action-head LayerNorm/residual-LN" \
  "experiments.leduc_poker.escher_action_head_layer_norm_residual_ablation.run" \
  --seed "${SMOKE_SEED}" \
  --variant-ids carry_forward_layer_norm_256_128_action_heads,deep_residual_layer_norm_256_256_128_action_heads \
  "${COMMON_ARGS[@]}" \
  --output-root "${SMOKE_OUTPUT_ROOT}/exp27_action_head_layer_norm_residual"

echo
echo "All recent ESCHER experiment smoke tests completed."
