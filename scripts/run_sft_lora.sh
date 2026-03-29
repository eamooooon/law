#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="sft-qwen2.5-4b-lr2e5-think"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

SWANLAB_PROJECT="law"
SWANLAB_API_KEY="zd9sy4txCTlXq4E3NkcGv"
export SWANLAB_PROJECT SWANLAB_API_KEY

ARGS=(
    # 路径
    --model_name_or_path "$PROJECT_ROOT/models/Qwen2.5-3B"
    --train_file_dir "$PROJECT_ROOT/data/think/"
    --output_dir "$PROJECT_ROOT/outputs-lora/$RUN_NAME"

    # 优化参数
    --num_train_epochs 1
    --learning_rate 2e-5
    --warmup_steps 5
    --weight_decay 0.05
    # --dataset_sample_ratio 0.3

    # 批大小
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 12
    --preprocessing_num_workers 8
    --model_max_length 4096

    # LoRA
    --use_peft True
    --target_modules all
    --lora_rank 32
    --lora_alpha 64
    --lora_dropout 0.1

    # 精度和显存
    --bf16
    --torch_dtype bfloat16
    --flash_attn True
    --gradient_checkpointing True

    # 训练评估
    --do_train
    --do_eval
    --seed 42
    # --max_train_samples 1000
    # --max_eval_samples 10

    # 日志和保存
    --report_to swanlab
    --run_name "$RUN_NAME"
    --logging_strategy steps
    --logging_steps 10
    --logging_first_step True
    --eval_steps 50
    --eval_strategy steps
    --save_steps 500
    --save_strategy steps
    --save_total_limit 10

    # DDP 参数
    --ddp_timeout 30000
    --ddp_find_unused_parameters False
)

echo "Using MASTER_PORT=$MASTER_PORT"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port "$MASTER_PORT" "$PROJECT_ROOT/src/sft.py" "${ARGS[@]}"