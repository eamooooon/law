#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASK_SUITE="${TASK_SUITE:-plan_b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-256}"

run_eval() {
  local model_dir="$1"
  local output_dir="$2"
  shift 2

  python "$PROJECT_ROOT/src/eval/eval_legalbench2.py" \
    --model_dir "$model_dir" \
    --output_dir "$output_dir" \
    --task_suite "$TASK_SUITE" \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_tokens "$EVAL_MAX_TOKENS" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    "$@"
}

cd "$PROJECT_ROOT"

run_eval "outputs/grpo-qwen2.5-3b-lr5e7" "results/grpo-qwen2.5-3b-lr5e7"


# run_eval "outputs/t1_full_mix" "results/t1_full_mix"
# run_eval "outputs/t2_lora_mix" "results/t2_lora_mix"
# run_eval "outputs/t3_full_en" "results/t3_full_en"
# run_eval "outputs/t4_lora_en" "results/t4_lora_en"


# run_eval "models/Qwen2.5-3B" "results/baseline"
# run_eval "outputs/sft-qwen2.5-3b-lr2e5-all-2" "results/sft"
# run_eval "outputs/sft-qwen2.5-3b-lr2e5-all-2" "results/sft-cot" --force_cot
# run_eval "outputs/sft-qwen2.5-3b-lr2e5-all-2" "results/sft-cot-rollout" \
#   --temperature 0.7 \
#   --top_p 0.9 \
#   --max_tokens 1024 \
#   --force_cot
