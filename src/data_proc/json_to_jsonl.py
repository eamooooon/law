#!/usr/bin/env python3
"""Convert JSON file(s) to JSONL.

Supported input JSON shapes:
1) top-level list: each element becomes one JSONL line
2) top-level dict: if contains key --list-key and value is list, list elements become lines;
   otherwise whole dict becomes one line
3) already line-delimited JSON content in .json file: each non-empty line is parsed and emitted

Examples:
  python src/data_proc/json_to_jsonl.py \
    --input-path datasets/raw/oyez_pretty.json \
    --output-path datasets/raw/oyez_pretty.jsonl

  python src/data_proc/json_to_jsonl.py \
    --input-path datasets/raw \
    --output-path datasets/raw_jsonl
"""

import argparse
import glob
import json
import os
from typing import Any, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON to JSONL.")
    parser.add_argument("--input-path", type=str, required=True, help="Input .json file or directory.")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output .jsonl file (when input is file) or output directory (when input is dir).",
    )
    parser.add_argument(
        "--list-key",
        type=str,
        default="",
        help="If top-level JSON is an object, extract records from this key if it is a list.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding for reading/writing.",
    )
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Escape non-ASCII characters in output JSONL.",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def list_json_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".json"):
            raise ValueError(f"Input file must end with .json: {input_path}")
        return [input_path]

    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "**", "*.json"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .json files found in directory: {input_path}")
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def try_parse_json_lines(text: str) -> List[Any]:
    records: List[Any] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {i} is not valid JSON: {exc}") from exc
    if not records:
        raise ValueError("Input has no non-empty JSON lines")
    return records


def extract_records(payload: Any, list_key: str) -> Iterable[Any]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if list_key:
            value = payload.get(list_key)
            if isinstance(value, list):
                return value
            raise ValueError(f"list_key '{list_key}' not found or not a list")
        return [payload]

    raise ValueError(f"Unsupported top-level JSON type: {type(payload).__name__}")


def resolve_output_file(input_file: str, output_path: str) -> str:
    if os.path.isdir(output_path) or output_path.endswith(os.sep):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.basename(input_file)
        stem, _ = os.path.splitext(base)
        return os.path.join(output_path, f"{stem}.jsonl")

    root, ext = os.path.splitext(output_path)
    if ext.lower() == ".jsonl":
        return output_path

    # If output_path does not exist and has no extension, treat as directory.
    if not ext and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.basename(input_file)
        stem, _ = os.path.splitext(base)
        return os.path.join(output_path, f"{stem}.jsonl")

    # Fallback: force .jsonl extension for file output.
    return f"{root}.jsonl"


def convert_one_file(input_file: str, output_file: str, list_key: str, encoding: str, ensure_ascii: bool) -> int:
    with open(input_file, "r", encoding=encoding) as fin:
        text = fin.read()

    try:
        payload = json.loads(text)
        records = list(extract_records(payload, list_key))
    except json.JSONDecodeError:
        records = try_parse_json_lines(text)

    ensure_parent(output_file)
    count = 0
    with open(output_file, "w", encoding=encoding) as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    files = list_json_files(args.input_path)

    if len(files) > 1 and os.path.isfile(args.output_path):
        raise ValueError("When input-path is a directory, output-path must be a directory.")

    total = 0
    for i, in_file in enumerate(files, start=1):
        out_file = resolve_output_file(in_file, args.output_path)
        rows = convert_one_file(
            input_file=in_file,
            output_file=out_file,
            list_key=args.list_key,
            encoding=args.encoding,
            ensure_ascii=args.ensure_ascii,
        )
        total += rows
        print(f"[{i}/{len(files)}] {in_file} -> {out_file} ({rows} rows)")

    print("=" * 68)
    print(f"Done. files={len(files)}, total_rows={total}")


if __name__ == "__main__":
    main()
