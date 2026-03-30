#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LAWBENCH_DATA_ROOT="${LAWBENCH_DATA_ROOT:-$PROJECT_ROOT/datasets/eval/LawBench/data}"
LAWBENCH_EVAL_ROOT="${LAWBENCH_EVAL_ROOT:-$PROJECT_ROOT/datasets/eval/LawBench/evaluation}"
SPLITS="${SPLITS:-zero_shot one_shot}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LAW_EVAL_MAX_TOKENS="${LAW_EVAL_MAX_TOKENS:-512}"
LAW_EVAL_TEMPERATURE="${LAW_EVAL_TEMPERATURE:-0.0}"
LAW_EVAL_TOP_P="${LAW_EVAL_TOP_P:-1.0}"
TASK_IDS="${TASK_IDS:-}"
MAX_TASKS="${MAX_TASKS:-0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是一个专业的中文法律助手。请严格按照题目要求作答，不要输出额外解释。}"
DISABLE_CHAT_TEMPLATE="${DISABLE_CHAT_TEMPLATE:-0}"

run_eval() {
  local model_dir="$1"
  local output_dir="$2"
  shift 2

  local system_name
  system_name="$(basename "$model_dir")"
  local pred_root="$PROJECT_ROOT/$output_dir/predictions"
  local metric_root="$PROJECT_ROOT/$output_dir"
  local extra_args=("$@")

  mkdir -p "$pred_root" "$metric_root"

  for split in $SPLITS; do
    local infer_args=()
    if [[ "$DISABLE_CHAT_TEMPLATE" == "1" ]]; then
      infer_args+=(--disable_chat_template)
    fi

    echo "========================================="
    echo "Running LawBench split: $split"
    echo "Model: $model_dir"
    echo "Output: $output_dir"
    echo "========================================="

    python "$PROJECT_ROOT/src/eval/eval_lawbench_native.py" \
      --model_dir "$model_dir" \
      --data_root "$LAWBENCH_DATA_ROOT" \
      --split "$split" \
      --output_root "$pred_root" \
      --system_name "$system_name" \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
      --max_tokens "$LAW_EVAL_MAX_TOKENS" \
      --temperature "$LAW_EVAL_TEMPERATURE" \
      --top_p "$LAW_EVAL_TOP_P" \
      --task_ids "$TASK_IDS" \
      --max_tasks "$MAX_TASKS" \
      --system_prompt "$SYSTEM_PROMPT" \
      "${infer_args[@]}" \
      "${extra_args[@]}"

    (
      cd "$LAWBENCH_EVAL_ROOT"
      python main.py \
        -i "$pred_root/$split" \
        -o "$metric_root/lawbench_${split}.csv"
    )
  done

  python "$PROJECT_ROOT/src/eval/summarize_lawbench_results.py" \
    --results_dir "$metric_root" \
    --data_root "$LAWBENCH_DATA_ROOT" \
    --model_dir "$model_dir"
}

cd "$PROJECT_ROOT"

# run_eval "outputs/t1_full_mix" "results/t1_full_mix_lawbench"

# run_eval "outputs/grpo-qwen2.5-3b-lr5e7" "results/grpo-qwen2.5-3b-lr5e7_lawbench"

run_eval "outputs/sft-qwen2.5-3b-lr2e5-comb" "results/sft-qwen2.5-3b-lr2e5-comb"
