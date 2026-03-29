#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH="datasets/think/echr_sft.jsonl"
OUTPUT_PATH="datasets/think/echr_sft_with_reasoning.jsonl"

python "$PROJECT_ROOT/src/data_proc/build_reasoning.py" \
  --input-path "$INPUT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --workers 4 \
  --chunk-size 16