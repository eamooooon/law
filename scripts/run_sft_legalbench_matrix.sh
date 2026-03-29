#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$PROJECT_ROOT/models/Qwen2.5-3B}"
DATA_MIX_DIR="${DATA_MIX_DIR:-$PROJECT_ROOT/data}"
DATA_EN_DIR="${DATA_EN_DIR:-$PROJECT_ROOT/data/sft-en}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-2}"
TASK_SUITE="${TASK_SUITE:-plan_b}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-256}"
TASK_LIMIT="${TASK_LIMIT:-10}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
SEED="${SEED:-42}"

SWANLAB_PROJECT="${SWANLAB_PROJECT:-law}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-zd9sy4txCTlXq4E3NkcGv}"
export SWANLAB_PROJECT SWANLAB_API_KEY

FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
LORA_OUTPUT_ROOT="${LORA_OUTPUT_ROOT:-$PROJECT_ROOT/outputs-lora}"
MERGED_OUTPUT_ROOT="${MERGED_OUTPUT_ROOT:-$PROJECT_ROOT/outputs-merged}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/results}"

T1_NAME="${T1_NAME:-t1_full_mix}"
T2_NAME="${T2_NAME:-t2_lora_mix}"
T3_NAME="${T3_NAME:-t3_full_en}"
T4_NAME="${T4_NAME:-t4_lora_en}"

run_train() {
    local run_name="$1"
    local train_dir="$2"
    local output_dir="$3"
    local use_peft="$4"

    local master_port
    master_port="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

    local -a args=(
        --model_name_or_path "$MODEL_NAME_OR_PATH"
        --train_file_dir "$train_dir"
        --output_dir "$output_dir"
        --num_train_epochs "$NUM_TRAIN_EPOCHS"
        --learning_rate "$LEARNING_RATE"
        --warmup_steps "$WARMUP_STEPS"
        --weight_decay "$WEIGHT_DECAY"
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
        --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS"
        --model_max_length "$MODEL_MAX_LENGTH"
        --use_peft "$use_peft"
        --bf16
        --torch_dtype bfloat16
        --flash_attn True
        --gradient_checkpointing True
        --do_train
        --do_eval
        --seed "$SEED"
        --report_to swanlab
        --run_name "$run_name"
        --logging_strategy steps
        --logging_steps 10
        --logging_first_step True
        --eval_steps 50
        --eval_strategy steps
        --save_steps 500
        --save_strategy steps
        --save_total_limit 10
        --ddp_timeout 30000
        --ddp_find_unused_parameters False
    )

    if [[ "$use_peft" == "True" ]]; then
        args+=(
            --target_modules all
            --lora_rank 32
            --lora_alpha 64
            --lora_dropout 0.1
        )
    fi

    echo "=================================================="
    echo "Train: $run_name"
    echo "Data : $train_dir"
    echo "Out  : $output_dir"
    echo "PEFT : $use_peft"
    echo "Port : $master_port"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
        torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$master_port" \
        "$PROJECT_ROOT/src/sft.py" "${args[@]}"
}

merge_lora_model() {
    local run_name="$1"
    local adapter_dir="$2"
    local merged_dir="$3"

    echo "=================================================="
    echo "Merge LoRA: $run_name"
    echo "Adapter: $adapter_dir"
    echo "Merged : $merged_dir"
    echo "=================================================="

    python "$PROJECT_ROOT/src/merge_lora.py" \
        --base_model "$MODEL_NAME_OR_PATH" \
        --adapter_model "$adapter_dir" \
        --output_dir "$merged_dir" \
        --torch_dtype bfloat16
}

run_eval() {
    local model_dir="$1"
    local output_dir="$2"

    echo "=================================================="
    echo "Eval model : $model_dir"
    echo "Eval out   : $output_dir"
    echo "=================================================="

    python "$PROJECT_ROOT/src/eval/eval_legalbench2.py" \
        --model_dir "$model_dir" \
        --output_dir "$output_dir" \
        --task_suite "$TASK_SUITE" \
        --temperature 0.0 \
        --top_p 1.0 \
        --max_tokens "$EVAL_MAX_TOKENS" \
        --tensor_parallel_size "$EVAL_TENSOR_PARALLEL_SIZE"
}

# mkdir -p "$FULL_OUTPUT_ROOT" "$LORA_OUTPUT_ROOT" "$MERGED_OUTPUT_ROOT" "$RESULT_ROOT"

# run_train "$T1_NAME" "$DATA_MIX_DIR" "$FULL_OUTPUT_ROOT/$T1_NAME" "False"
# run_eval "$FULL_OUTPUT_ROOT/$T1_NAME" "$RESULT_ROOT/$T1_NAME"

# run_train "$T2_NAME" "$DATA_MIX_DIR" "$LORA_OUTPUT_ROOT/$T2_NAME" "True"
# merge_lora_model "$T2_NAME" "$LORA_OUTPUT_ROOT/$T2_NAME" "$MERGED_OUTPUT_ROOT/$T2_NAME"
# run_eval "$MERGED_OUTPUT_ROOT/$T2_NAME" "$RESULT_ROOT/$T2_NAME"

run_train "$T3_NAME" "$DATA_EN_DIR" "$FULL_OUTPUT_ROOT/$T3_NAME" "False"
run_eval "$FULL_OUTPUT_ROOT/$T3_NAME" "$RESULT_ROOT/$T3_NAME"

run_train "$T4_NAME" "$DATA_EN_DIR" "$LORA_OUTPUT_ROOT/$T4_NAME" "True"
merge_lora_model "$T4_NAME" "$LORA_OUTPUT_ROOT/$T4_NAME" "$MERGED_OUTPUT_ROOT/$T4_NAME"
run_eval "$MERGED_OUTPUT_ROOT/$T4_NAME" "$RESULT_ROOT/$T4_NAME"

echo "=================================================="
echo "All T1-T4 experiments completed."
echo "Results root: $RESULT_ROOT"
echo "=================================================="
