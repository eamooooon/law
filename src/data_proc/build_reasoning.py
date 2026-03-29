#!/usr/bin/env python3
"""
Generate missing reasoning for ECHR SFT JSONL records with an external LLM.

For each record whose `reasoning` field is empty, this script asks an LLM to produce:
1) `reasoning`: first-person legal analysis narrative (task-level rationale, not hidden model chain-of-thought)
2) `output_model`: first-person final answer

The script keeps original fields and appends/updates `output_model`.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple
from urllib import request
from urllib.parse import urlparse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_API_URL = "http://101.33.45.201:8317"
DEFAULT_API_KEY = "sk-6Lw43uf1fvahDCTLpg0lVOQaLsVPrs3rN3XDtlCzdQFTr"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_REASONING_CHARS = 1000

SYSTEM_PROMPT = (
    "You are a legal analyst handling ECHR case summaries for training data construction. "
    "I must speak in first person only, using a judge-like legal reasoning style. "
    "I must avoid generic templates and produce fact-grounded analysis specific to each case. "
    "I must not use third-person self-reference such as 'the model', 'this AI', or 'the assistant'."
)

PROMPT_TEMPLATE = """You will receive one ECHR training sample with fields instruction, input, and gold output.

Task:
- Produce missing reasoning and answer in FIRST PERSON.
- The reasoning must be a case-specific legal analysis, not a generic template.
- The answer must also be in first person and should directly conclude the judgment.
- Do not reveal hidden internal chain-of-thought; provide only task-relevant reasoning steps suitable for training data.
- Do not mention 'as an AI', 'the model thinks', or any third-person self-reference.

Reasoning quality requirements:
1) I must identify the key legal issue(s) under the Convention from the case facts.
2) I must cite concrete factual anchors from the input (timeline, delays, procedure, remedies, enforcement, detention, etc.).
3) I must explain why those facts satisfy or fail the legal threshold (for example: reasonable time, effective remedy, access to court, peaceful enjoyment of possessions).
4) Where appropriate, I should briefly acknowledge a plausible counter-consideration and explain why it does or does not change my conclusion.
5) My reasoning should be substantive and varied in expression, not repetitive boilerplate.

Style constraints:
- reasoning: one coherent first-person narrative paragraph. Length is flexible; depth and legal sufficiency are the priority.
- I should write enough detail to make the legal path clear (often longer than a short template), but I must avoid filler and repetitive phrasing.
- If the case facts are complex (multiple procedural stages, remedies, or articles), I should provide correspondingly fuller analysis.
- reasoning must not exceed 1000 characters.
- output_model: concise first-person final decision summary, aligned with but not copied verbatim from gold_output.
- Keep legal article references accurate and consistent with the conclusion.

Return STRICT JSON only:
{{
  "reasoning": "...",
  "output_model": "..."
}}

Input record:
instruction: {instruction}
input: {input_text}
gold_output: {gold_output}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build missing reasoning/output_model for ECHR JSONL with LLM.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input JSONL file path or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSONL file path or output directory.",
    )
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Chat completions endpoint URL.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max output tokens.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if reasoning already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N valid records (0 means all).")
    parser.add_argument("--start", type=int, default=0, help="Skip first N valid records before processing.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout (seconds).")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for each failed LLM call.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers for API generation.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Records per scheduling chunk. 0 means auto (workers * 4).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file by continuing after already written lines.",
    )
    parser.add_argument(
        "--max-reasoning-chars",
        type=int,
        default=DEFAULT_MAX_REASONING_CHARS,
        help="Hard upper bound for reasoning length in characters.",
    )
    return parser.parse_args()


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def count_nonempty_lines(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                count += 1
    return count


def count_input_rows(file_path: str) -> int:
    """Count non-empty lines as the file total rows for progress display."""
    return count_nonempty_lines(file_path)


def build_output_file_path(input_file: str, output_path: str) -> str:
    """Resolve output file path when output_path can be file or directory."""
    if output_path.endswith(os.sep) or os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.basename(input_file)
        stem, ext = os.path.splitext(base)
        return os.path.join(output_path, f"{stem}_with_reasoning{ext or '.jsonl'}")

    # If output_path doesn't exist yet and has no extension, treat as directory.
    root, ext = os.path.splitext(output_path)
    if not ext and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.basename(input_file)
        stem, in_ext = os.path.splitext(base)
        return os.path.join(output_path, f"{stem}_with_reasoning{in_ext or '.jsonl'}")

    return output_path


def list_input_files(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
        if files:
            return files
        raise FileNotFoundError(f"No .jsonl files found in input directory: {input_path}")
    raise FileNotFoundError(f"Input path not found: {input_path}")


def needs_generation(record: Dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return True
    reasoning = str(record.get("reasoning", "")).strip()
    output_model = str(record.get("output_model", "")).strip()
    return (not reasoning) or (not output_model)


def sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def shorten_reasoning(reasoning: str, max_chars: int) -> str:
    """Ensure reasoning stays within max_chars while preserving readability."""
    if max_chars <= 0 or len(reasoning) <= max_chars:
        return reasoning

    trimmed = reasoning[:max_chars].rstrip()

    # Prefer ending at a sentence boundary if possible.
    punctuations = ["。", ".", "!", "?", "；", ";"]
    cut_positions = [trimmed.rfind(p) for p in punctuations]
    cut = max(cut_positions)
    if cut >= int(max_chars * 0.7):
        return trimmed[: cut + 1].rstrip()

    return trimmed


def normalize_chat_url(api_url: str) -> str:
    """Allow base URL input and normalize to chat-completions endpoint."""
    raw = (api_url or "").strip()
    if not raw:
        raise ValueError("api_url is empty")

    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw

    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")

    if path.endswith("/v1/chat/completions"):
        return raw

    if path == "":
        return raw.rstrip("/") + "/v1/chat/completions"

    return raw


def build_user_prompt(record: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=sanitize_text(record.get("instruction", "")),
        input_text=sanitize_text(record.get("input", "")),
        gold_output=sanitize_text(record.get("output", "")),
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    # Direct JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Response is not valid JSON object")


def call_chat_completions(
    api_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    chat_url = normalize_chat_url(api_url)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(chat_url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    parsed = json.loads(raw)
    choices = parsed.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in response: {raw[:500]}")

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if not content:
        raise ValueError(f"Empty content in response: {raw[:500]}")
    return content


def generate_for_record(record: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str]:
    prompt = build_user_prompt(record)
    last_err: Optional[Exception] = None

    for attempt in range(1, args.retries + 1):
        try:
            content = call_chat_completions(
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                user_prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            parsed = extract_json_from_text(content)

            reasoning = sanitize_text(parsed.get("reasoning", ""))
            output_model = sanitize_text(parsed.get("output_model", ""))
            reasoning = shorten_reasoning(reasoning, args.max_reasoning_chars)

            if not reasoning or not output_model:
                raise ValueError("Missing 'reasoning' or 'output_model' in model output")

            return reasoning, output_model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < args.retries:
                backoff = min(2 ** (attempt - 1), 8)
                print(f"[WARN] attempt {attempt}/{args.retries} failed: {exc}; retry in {backoff}s", flush=True)
                time.sleep(backoff)
            else:
                break

    raise RuntimeError(f"Generation failed after {args.retries} attempts: {last_err}")


def process_file(args: argparse.Namespace, input_file: str, output_file: str) -> Dict[str, int]:
    ensure_parent_dir(output_file)

    file_total_rows = count_input_rows(input_file)
    print(f"[INFO] current file total rows: {file_total_rows}", flush=True)
    run_start_time = time.time()

    resume_lines = count_nonempty_lines(output_file) if args.resume else 0
    if args.resume and resume_lines > 0:
        print(f"[INFO] resume enabled: found {resume_lines} existing lines in output", flush=True)

    write_mode = "a" if args.resume and os.path.exists(output_file) else "w"

    total = 0
    selected = 0
    generated = 0
    skipped = 0
    failed = 0

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    chunk_size = args.chunk_size if args.chunk_size > 0 else max(args.workers * 4, 4)

    def flush_chunk(chunk: list[Dict[str, Any]], fout_handle, executor: Optional[ThreadPoolExecutor]) -> None:
        nonlocal generated, failed
        if not chunk:
            return

        if executor is not None:
            for item in chunk:
                if item["needs_gen"]:
                    item["future"] = executor.submit(generate_for_record, item["record"], args)

        try:
            for item in chunk:
                record = item["record"]
                line_no = item["line_no"]
                rec_id = item["rec_id"]

                if item["needs_gen"]:
                    try:
                        if executor is not None:
                            reasoning, output_model = item["future"].result()
                        else:
                            reasoning, output_model = generate_for_record(record, args)
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        print(f"[ERROR] {rec_id}: {exc}", file=sys.stderr, flush=True)
                        raise RuntimeError(
                            f"Generation aborted at {rec_id} (line {line_no}). "
                            "No further records were written after this failure."
                        ) from exc

                    record["reasoning"] = reasoning
                    record["output_model"] = output_model
                    generated += 1

                    if generated % 100 == 0:
                        elapsed = max(time.time() - run_start_time, 1e-6)
                        speed = generated / elapsed
                        print(
                            f"[PROGRESS] generated={generated} / total={file_total_rows} "
                            f"(line={line_no}) elapsed={elapsed:.1f}s speed={speed:.2f} rows/s",
                            flush=True,
                        )

                    if args.sleep > 0:
                        time.sleep(args.sleep)
                else:
                    skipped_local = item.get("skipped_local", 0)
                    if skipped_local:
                        pass

                fout_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            if executor is not None:
                for item in chunk:
                    fut = item.get("future")
                    if fut is not None and not fut.done():
                        fut.cancel()
            raise

    executor: Optional[ThreadPoolExecutor] = None
    if args.workers > 1:
        print(f"[INFO] concurrency enabled: workers={args.workers}, chunk_size={chunk_size}", flush=True)
        executor = ThreadPoolExecutor(max_workers=args.workers)
    else:
        print("[INFO] concurrency disabled: workers=1", flush=True)

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, write_mode, encoding="utf-8") as fout:
        try:
            chunk: list[Dict[str, Any]] = []

            for line_no, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                total += 1

                # Resume: skip lines that have already been written to output.
                if total <= resume_lines:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    failed += 1
                    print(f"[ERROR] line {line_no}: invalid JSON: {exc}", file=sys.stderr, flush=True)
                    continue

                if not isinstance(record, dict):
                    failed += 1
                    print(f"[ERROR] line {line_no}: JSON is not object", file=sys.stderr, flush=True)
                    continue

                rec_id = sanitize_text(record.get("id", f"line_{line_no}"))

                if total <= args.start:
                    skipped += 1
                    chunk.append({"record": record, "line_no": line_no, "rec_id": rec_id, "needs_gen": False})
                elif args.limit > 0 and selected >= args.limit:
                    skipped += 1
                    chunk.append({"record": record, "line_no": line_no, "rec_id": rec_id, "needs_gen": False})
                elif not needs_generation(record, args.overwrite):
                    skipped += 1
                    chunk.append({"record": record, "line_no": line_no, "rec_id": rec_id, "needs_gen": False})
                else:
                    selected += 1
                    chunk.append({"record": record, "line_no": line_no, "rec_id": rec_id, "needs_gen": True})

                if len(chunk) >= chunk_size:
                    flush_chunk(chunk, fout, executor)
                    chunk = []

            if chunk:
                flush_chunk(chunk, fout, executor)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    print("=" * 70)
    print("Done.")
    print(f"Input file   : {input_file}")
    print(f"Output file  : {output_file}")
    print(f"Total rows   : {total}")
    print(f"Selected rows: {selected}")
    print(f"Generated    : {generated}")
    print(f"Skipped      : {skipped}")
    print(f"Failed       : {failed}")

    return {
        "total": total,
        "selected": selected,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
    }


def process_path(args: argparse.Namespace) -> None:
    input_files = list_input_files(args.input_path)

    if len(input_files) > 1 and os.path.isfile(args.output_path):
        raise ValueError("When input is a directory, --output-path must be a directory.")

    grand_total = 0
    grand_selected = 0
    grand_generated = 0
    grand_skipped = 0
    grand_failed = 0

    for idx, input_file in enumerate(input_files, start=1):
        output_file = build_output_file_path(input_file, args.output_path)
        print(f"\n[{idx}/{len(input_files)}] Processing: {input_file}")
        stats = process_file(args, input_file, output_file)
        grand_total += stats["total"]
        grand_selected += stats["selected"]
        grand_generated += stats["generated"]
        grand_skipped += stats["skipped"]
        grand_failed += stats["failed"]

    if len(input_files) > 1:
        print("\n" + "#" * 70)
        print("All files done.")
        print(f"Files processed: {len(input_files)}")
        print(f"Total rows   : {grand_total}")
        print(f"Selected rows: {grand_selected}")
        print(f"Generated    : {grand_generated}")
        print(f"Skipped      : {grand_skipped}")
        print(f"Failed       : {grand_failed}")


if __name__ == "__main__":
    cli_args = parse_args()
    process_path(cli_args)
