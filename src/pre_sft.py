#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为法律 SFT 构建多任务混训数据。

设计目标：
1. 不修改现有 law/src/sft.py。
2. 先把不同来源数据按任务类型分组。
3. 按任务分层切分 train/validation。
4. 按任务权重做重采样，避免某一类任务淹没其它任务。
5. 给每条样本补充 task_name / system_prompt / mixed_instruction，便于现有 sft.py 直接训练。

输入数据格式（JSONL）：
    {"instruction": "...", "input": "...", "reasoning": "...", "output": "..."}

输出数据格式（JSONL）：
    保留原字段，并额外写入：
    - task_name
    - system_prompt
    - mixed_instruction
    - dataset_name
    - source_file

训练时建议：
    让 sft.py 使用 mixed_instruction 作为 instruction 列，或者在本脚本里直接覆盖 instruction。
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from glob import glob
from typing import Dict, Iterable, List, Tuple


TASK_SYSTEM_PROMPTS = {
    "case_classification": "你是严谨的中文法律助手。请根据案情做定性分析，给出简洁、准确、结构稳定的结论。",
    "reading_qa": "你是严谨的中文法律助手。请只根据给定案件事实回答问题，不扩写无关内容。",
    "evidence_qa": "你是严谨的中文法律助手。请先抓住支持结论的关键事实，再给出简洁答案。",
    "mcq": "你是严谨的中文法律助手。请基于法律规则完成选择题，并保持答案格式稳定。",
    "case_summary": "你是严谨的中文法律助手。请压缩冗长司法文本，只保留事实主线、争议焦点和裁判结果。",
    "argument_summary": "你是严谨的法律摘要助手。请将法律论证提炼为高信息密度的标题或摘要。",
    "other": "你是严谨的法律助手。请依据输入完成任务，保持准确、简洁、格式稳定。",
}

TASK_DISPLAY_NAMES = {
    "case_classification": "刑事定性与量刑",
    "reading_qa": "法律阅读问答",
    "evidence_qa": "证据支持问答",
    "mcq": "法律选择题",
    "case_summary": "司法文书摘要",
    "argument_summary": "法律论证摘要",
    "other": "其他法律任务",
}

DEFAULT_TASK_WEIGHTS = {
    "case_classification": 1.0,
    "reading_qa": 1.0,
    "evidence_qa": 1.0,
    "mcq": 1.0,
    "case_summary": 0.7,
    "argument_summary": 0.7,
    "other": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multi-task mixed SFT jsonl.")
    parser.add_argument(
        "--input-paths",
        type=str,
        required=True,
        help="逗号分隔的输入路径；支持目录或单个 jsonl 文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录，将写出 train.jsonl / validation.jsonl / metadata.json",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="每个任务内部的验证集切分比例。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--weight-overrides",
        type=str,
        default="",
        help="覆盖默认任务权重，例如 mcq=1.2,case_summary=0.5",
    )
    parser.add_argument(
        "--max-oversample-factor",
        type=float,
        default=3.0,
        help="单任务最多放大倍数，防止小数据集被过度重复。",
    )
    parser.add_argument(
        "--instruction-mode",
        type=str,
        default="replace",
        choices=["prefix", "replace"],
        help="prefix: 保留原 instruction 并增加任务标签；replace: 用 mixed_instruction 覆盖 instruction。",
    )
    parser.add_argument(
        "--drop-empty-output",
        action="store_true",
        help="丢弃 output 为空的样本。",
    )
    return parser.parse_args()


def collect_jsonl_files(input_paths: str) -> List[str]:
    files: List[str] = []
    for raw_item in input_paths.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if os.path.isfile(item):
            files.append(item)
            continue
        files.extend(sorted(glob(os.path.join(item, "**", "*.jsonl"), recursive=True)))
    return sorted(set(files))


def parse_weight_overrides(raw_text: str) -> Dict[str, float]:
    if not raw_text.strip():
        return {}
    overrides: Dict[str, float] = {}
    for chunk in raw_text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid weight override: {part}")
        key, value = part.split("=", 1)
        overrides[key.strip()] = float(value.strip())
    return overrides


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def detect_task_name(record: dict, source_file: str) -> str:
    source = str(record.get("source", "")).lower()
    instruction = str(record.get("instruction", "")).lower()
    file_name = os.path.basename(source_file).lower()
    input_text = str(record.get("input", ""))
    output_text = str(record.get("output", ""))

    if "jec-qa" in source or "jec_qa" in source or "选择题" in instruction:
        return "mcq"
    if "ydlj" in source or "按步骤提取支持证据" in instruction:
        return "evidence_qa"
    if "sfzy" in source or "案情摘要" in instruction or "提炼事实焦点" in instruction:
        return "case_summary"
    if "cail2018" in source or "罪名" in instruction or "刑期" in instruction:
        return "case_classification"
    if "cail2019" in source or "找出问题的正确答案" in instruction:
        return "reading_qa"
    if "briefme" in file_name or "section heading" in instruction:
        return "argument_summary"
    if "casesumm" in file_name or "summary" in instruction:
        return "case_summary"
    if "question" in input_text.lower() and output_text.strip():
        return "reading_qa"
    return "other"


def build_mixed_instruction(record: dict, task_name: str) -> str:
    original = str(record.get("instruction", "")).strip()
    task_label = TASK_DISPLAY_NAMES.get(task_name, task_name)
    if original:
        return f"[任务类型] {task_label}\n{original}"
    return f"[任务类型] {task_label}"


def validate_record(record: dict, drop_empty_output: bool) -> bool:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output_text = str(record.get("output", "")).strip()
    if not instruction or not input_text:
        return False
    if drop_empty_output and not output_text:
        return False
    return True


def split_by_task(records: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in records:
        grouped[row["task_name"]].append(row)

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    rng = random.Random(seed)

    for task_name, items in grouped.items():
        rng.shuffle(items)
        if len(items) == 1:
            train_rows.extend(items)
            continue
        val_count = max(1, int(round(len(items) * val_ratio)))
        if val_count >= len(items):
            val_count = len(items) - 1
        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def rebalance_train(
    train_rows: List[dict],
    weights: Dict[str, float],
    max_oversample_factor: float,
    seed: int,
) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in train_rows:
        grouped[row["task_name"]].append(row)

    if not grouped:
        return []

    rng = random.Random(seed)
    max_count = max(len(items) for items in grouped.values())
    mixed_rows: List[dict] = []

    for task_name, items in grouped.items():
        weight = weights.get(task_name, DEFAULT_TASK_WEIGHTS.get(task_name, 1.0))
        target_count = int(round(max_count * weight))
        max_allowed = max(1, int(math.ceil(len(items) * max_oversample_factor)))
        target_count = max(1, min(target_count, max_allowed))

        if target_count <= len(items):
            selected = rng.sample(items, target_count)
        else:
            selected = list(items)
            extra_needed = target_count - len(items)
            selected.extend(rng.choices(items, k=extra_needed))

        mixed_rows.extend(selected)

    rng.shuffle(mixed_rows)
    return mixed_rows


def summarize_counts(rows: Iterable[dict]) -> Dict[str, int]:
    return dict(Counter(row["task_name"] for row in rows))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    files = collect_jsonl_files(args.input_paths)
    if not files:
        raise FileNotFoundError(f"No jsonl files found from: {args.input_paths}")

    weights = dict(DEFAULT_TASK_WEIGHTS)
    weights.update(parse_weight_overrides(args.weight_overrides))

    all_rows: List[dict] = []
    skipped = 0
    source_file_counts: Dict[str, int] = {}

    for path in files:
        rows = load_jsonl(path)
        source_file_counts[path] = len(rows)
        for row in rows:
            if not validate_record(row, args.drop_empty_output):
                skipped += 1
                continue
            item = dict(row)
            task_name = detect_task_name(item, path)
            item["task_name"] = task_name
            item["dataset_name"] = str(item.get("source", "")).strip() or os.path.basename(path)
            item["source_file"] = path
            item["system_prompt"] = TASK_SYSTEM_PROMPTS.get(task_name, TASK_SYSTEM_PROMPTS["other"])
            item["mixed_instruction"] = build_mixed_instruction(item, task_name)
            if args.instruction_mode == "replace":
                item["instruction"] = item["mixed_instruction"]
            all_rows.append(item)

    if not all_rows:
        raise ValueError("No valid rows found after filtering.")

    train_rows, val_rows = split_by_task(all_rows, args.val_ratio, args.seed)
    mixed_train_rows = rebalance_train(
        train_rows=train_rows,
        weights=weights,
        max_oversample_factor=args.max_oversample_factor,
        seed=args.seed,
    )

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "validation.jsonl")
    meta_path = os.path.join(args.output_dir, "metadata.json")

    write_jsonl(train_path, mixed_train_rows)
    write_jsonl(val_path, val_rows)

    metadata = {
        "input_paths": files,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "instruction_mode": args.instruction_mode,
        "max_oversample_factor": args.max_oversample_factor,
        "task_weights": weights,
        "source_file_counts": source_file_counts,
        "skipped_records": skipped,
        "raw_total_records": len(all_rows),
        "train_counts_before_rebalance": summarize_counts(train_rows),
        "train_counts_after_rebalance": summarize_counts(mixed_train_rows),
        "validation_counts": summarize_counts(val_rows),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Prepared multi-task SFT data:")
    print(f"  train      : {train_path}")
    print(f"  validation : {val_path}")
    print(f"  metadata   : {meta_path}")
    print("Train counts after rebalance:")
    for task_name, count in sorted(metadata["train_counts_after_rebalance"].items()):
        print(f"  - {task_name}: {count}")


if __name__ == "__main__":
    main()
