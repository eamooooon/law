#!/bin/bash

# Legal GRPO training entrypoint.
# Default behavior:
# - read local json/jsonl files from data/grpo
# - use input as the user prompt
# - reward format(only <think>) + think length + answer overlength penalty
# - keep LLM judge disabled until API settings are provided

MODEL_PATH="outputs/t1_full_mix"
MODEL_NAME="grpo-qwen2.5-3b-lr5e7"
RUN_NAME="grpo-${MODEL_NAME}"
OUTPUT_DIR="outputs/${RUN_NAME}"
DEEPSPEED_CONFIG="config/grpo_zero2.json"

SWANLAB_PROJECT="law"
SWANLAB_API_KEY="zd9sy4txCTlXq4E3NkcGv"
export SWANLAB_PROJECT SWANLAB_API_KEY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 src/grpo_training.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_file_dir data/grpo \
    --use_only_input_prompt True \
    --train_samples -1 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 7 \
    --output_dir ${OUTPUT_DIR} \
    --dtype bfloat16 \
    --bf16 True \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --report_to swanlab \
    --run_name ${RUN_NAME} \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    \
    --format_reward_weight 1.0 \
    --think_length_cap 2048 \
    --think_length_reward_max 1.0 \
    --answer_soft_limit 512 \
    --answer_hard_limit 1024 \
    --answer_over_limit_penalty_max 1.0 \
    --judge_enabled False \
    --judge_score_threshold 0.6 \
    \
    --use_peft False \
    \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_generations 2 \
    --gradient_accumulation_steps 8 \
    --max_completion_length 2048

echo "训练完成!"
