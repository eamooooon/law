#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH="datasets/think/cail2018_5000.jsonl"
OUTPUT_PATH="datasets/think/cail2018_5000_with_reasoning.jsonl"

python "$PROJECT_ROOT/src/data_proc/build_reasoning_zh.py" \
  --input-path "$INPUT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --workers 4 \
  --chunk-size 16 \
  --resume