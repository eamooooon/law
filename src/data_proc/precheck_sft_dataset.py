#!/usr/bin/env python3
"""
Precheck SFT dataset compatibility with src/sft.py preprocessing rules.

Checks per sample:
1) required fields: instruction/input/reasoning/output are present and non-empty
2) build assistant target as <think>reasoning</think> + output
3) tokenize with tokenizer.apply_chat_template
4) truncate to max_length
5) reject sample if prompt_len >= len(full_ids) after truncation

This script helps explain why training data may become empty after preprocessing.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from glob import glob
from statistics import mean
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precheck SFT dataset for src/sft.py.")
    parser.add_argument("--input-path", type=str, required=True, help="JSON/JSONL file or directory.")
    parser.add_argument("--model-name-or-path", type=str, default="models/Qwen2.5-3B", help="Model/tokenizer path.")
    parser.add_argument("--tokenizer-name-or-path", type=str, default="models/Qwen2.5-3B", help="Optional tokenizer path.")
    parser.add_argument("--max-length", type=int, default=4096, help="Same as --model_max_length in training.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Default system prompt when sample has no system_prompt.",
    )
    parser.add_argument("--sample-limit", type=int, default=0, help="Check only first N rows (0 means all).")
    parser.add_argument("--show-examples", type=int, default=3, help="Show first K examples per failure reason.")
    return parser.parse_args()


def collect_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(glob(os.path.join(input_path, "**", "*.json"), recursive=True))
        files += sorted(glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True))
        return files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def wrap_with_think_tag(text: str) -> str:
    content = text.strip()
    if content.startswith("<think>") and content.endswith("</think>"):
        content = content[len("<think>") : -len("</think>")].strip()
    return f"<think>\n{content}\n</think>"


def get_system_prompt(record: Dict, default_prompt: str) -> str:
    sp = record.get("system_prompt", "")
    if sp is None:
        return default_prompt
    sp = str(sp).strip()
    return sp if sp else default_prompt


def summarize_lengths(values: List[int]) -> str:
    if not values:
        return "n=0"
    vals = sorted(values)
    n = len(vals)
    p50 = vals[n // 2]
    p90 = vals[int(n * 0.9)] if n > 1 else vals[0]
    p99 = vals[int(n * 0.99)] if n > 1 else vals[0]
    return f"n={n}, min={vals[0]}, mean={mean(vals):.1f}, p50={p50}, p90={p90}, p99={p99}, max={vals[-1]}"


def normalize_token_ids(tokenized) -> List[int]:
    """Normalize apply_chat_template return values to a plain list[int]."""
    if isinstance(tokenized, dict):
        ids = tokenized.get("input_ids")
    elif hasattr(tokenized, "input_ids"):
        ids = tokenized.input_ids
    else:
        ids = tokenized

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    if isinstance(ids, tuple):
        ids = list(ids)

    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]

    if not isinstance(ids, list):
        raise TypeError(f"Unexpected tokenized type: {type(tokenized)}")
    return ids


def main() -> None:
    args = parse_args()

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE

    files = collect_files(args.input_path)
    if not files:
        raise ValueError("No json/jsonl files found.")

    required_fields = ["instruction", "input", "reasoning", "output"]
    reasons = Counter()
    examples_by_reason = defaultdict(list)

    total_rows = 0
    parsed_rows = 0
    passed_rows = 0

    full_len_before_trunc = []
    full_len_after_trunc = []
    prompt_len_list = []
    target_len_list = []

    stop = False

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                if stop:
                    break

                line = line.strip()
                if not line:
                    continue

                total_rows += 1
                if args.sample_limit > 0 and total_rows > args.sample_limit:
                    stop = True
                    break

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    reasons["invalid_json"] += 1
                    if len(examples_by_reason["invalid_json"]) < args.show_examples:
                        examples_by_reason["invalid_json"].append(f"{file_path}:{line_no}")
                    continue

                if not isinstance(record, dict):
                    reasons["not_object"] += 1
                    if len(examples_by_reason["not_object"]) < args.show_examples:
                        examples_by_reason["not_object"].append(f"{file_path}:{line_no}")
                    continue

                parsed_rows += 1

                missing = [k for k in required_fields if k not in record]
                if missing:
                    key = f"missing_fields:{','.join(missing)}"
                    reasons[key] += 1
                    if len(examples_by_reason[key]) < args.show_examples:
                        examples_by_reason[key].append(f"{file_path}:{line_no}")
                    continue

                instruction = str(record.get("instruction", "")).strip()
                input_text = str(record.get("input", "")).strip()
                reasoning = str(record.get("reasoning", "")).strip()
                output = str(record.get("output", "")).strip()

                if not instruction or not input_text or not reasoning or not output:
                    reasons["empty_required_field"] += 1
                    if len(examples_by_reason["empty_required_field"]) < args.show_examples:
                        rid = str(record.get("id", f"{file_path}:{line_no}"))
                        examples_by_reason["empty_required_field"].append(rid)
                    continue

                assistant_text = f"{wrap_with_think_tag(reasoning)}\n{output}"
                user_query = f"{instruction}\n{input_text}"
                system_prompt = get_system_prompt(record, args.system_prompt)

                messages_full = []
                if system_prompt:
                    messages_full.append({"role": "system", "content": system_prompt})
                messages_full.append({"role": "user", "content": user_query})
                messages_full.append({"role": "assistant", "content": assistant_text})

                messages_prompt = []
                if system_prompt:
                    messages_prompt.append({"role": "system", "content": system_prompt})
                messages_prompt.append({"role": "user", "content": user_query})

                try:
                    full_ids = tokenizer.apply_chat_template(
                        messages_full, tokenize=True, add_generation_prompt=False
                    )
                    prompt_ids = tokenizer.apply_chat_template(
                        messages_prompt, tokenize=True, add_generation_prompt=True
                    )
                    full_ids = normalize_token_ids(full_ids)
                    prompt_ids = normalize_token_ids(prompt_ids)
                except Exception as exc:  # noqa: BLE001
                    reasons["tokenize_error"] += 1
                    if len(examples_by_reason["tokenize_error"]) < args.show_examples:
                        rid = str(record.get("id", f"{file_path}:{line_no}"))
                        examples_by_reason["tokenize_error"].append(f"{rid} ({exc})")
                    continue

                full_len_before_trunc.append(len(full_ids))
                if len(full_ids) > args.max_length:
                    full_ids = full_ids[: args.max_length]

                prompt_len = min(len(prompt_ids), len(full_ids))
                if prompt_len >= len(full_ids):
                    reasons["truncated_prompt_only"] += 1
                    if len(examples_by_reason["truncated_prompt_only"]) < args.show_examples:
                        rid = str(record.get("id", f"{file_path}:{line_no}"))
                        examples_by_reason["truncated_prompt_only"].append(rid)
                    continue

                full_len_after_trunc.append(len(full_ids))
                prompt_len_list.append(prompt_len)
                target_len_list.append(len(full_ids) - prompt_len)
                passed_rows += 1

    print("=" * 78)
    print("SFT Dataset Precheck Report")
    print(f"files_checked: {len(files)}")
    print(f"total_rows: {total_rows}")
    print(f"parsed_rows: {parsed_rows}")
    print(f"passed_rows: {passed_rows}")
    print(f"failed_rows: {parsed_rows - passed_rows}")
    print("-" * 78)
    print(f"full_len_before_trunc: {summarize_lengths(full_len_before_trunc)}")
    print(f"full_len_after_trunc : {summarize_lengths(full_len_after_trunc)}")
    print(f"prompt_len           : {summarize_lengths(prompt_len_list)}")
    print(f"target_len           : {summarize_lengths(target_len_list)}")
    print("-" * 78)

    if reasons:
        print("Failure reason counts:")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")
            exs = examples_by_reason.get(reason, [])
            for ex in exs:
                print(f"    - {ex}")
    else:
        print("No failures found.")

    print("=" * 78)

    if passed_rows == 0:
        raise SystemExit("No valid samples passed precheck. Please fix dataset or preprocessing settings.")


if __name__ == "__main__":
    main()
