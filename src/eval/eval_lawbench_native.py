import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native LawBench inference without OpenCompass")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory for vLLM loading.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/eval/LawBench/data",
        help="LawBench data root containing zero_shot and one_shot subfolders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "one_shot"],
        help="LawBench split to run.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/lawbench_native",
        help="Root directory for prediction outputs.",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        default="",
        help="Folder name used by the native LawBench evaluator. Defaults to model basename.",
    )
    parser.add_argument("--task_ids", type=str, default="", help="Comma-separated LawBench task IDs, e.g. 1-1,1-2.")
    parser.add_argument("--max_tasks", type=int, default=0, help="Optional limit on number of tasks.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum generated tokens per sample.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="你是一个专业的中文法律助手。请严格按照题目要求作答，不要输出额外解释。",
        help="Optional system prompt. Set to empty string to disable.",
    )
    parser.add_argument(
        "--disable_chat_template",
        action="store_true",
        help="Use raw concatenated prompts instead of tokenizer chat template.",
    )
    return parser.parse_args()


def load_task_file(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_origin_prompt(item: Dict) -> str:
    instruction = str(item.get("instruction", "")).strip()
    question = str(item.get("question", "")).strip()
    if instruction and question:
        return f"{instruction}\n{question}"
    return instruction or question


def build_generation_prompt(tokenizer, origin_prompt: str, system_prompt: str, disable_chat_template: bool) -> str:
    if disable_chat_template:
        if system_prompt:
            return f"{system_prompt}\n\n{origin_prompt}"
        return origin_prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": origin_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.rstrip("/")
    system_name = args.system_name.strip() or os.path.basename(model_dir)
    split_dir = Path(args.data_root) / args.split
    output_dir = Path(args.output_root) / args.split / system_name
    output_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(split_dir.glob("*.json"))
    if args.task_ids.strip():
        selected = {task.strip() for task in args.task_ids.split(",") if task.strip()}
        task_files = [path for path in task_files if path.stem in selected]
    elif args.max_tasks > 0:
        task_files = task_files[: args.max_tasks]

    print("=========================================")
    print("LawBench native inference")
    print("=========================================")
    print(f"Model: {model_dir}")
    print(f"Split: {args.split}")
    print(f"Tasks: {len(task_files)}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for task_path in task_files:
        task_id = task_path.stem
        print(f"\nProcessing {task_id}")
        records = load_task_file(task_path)

        origin_prompts = [build_origin_prompt(item) for item in records]
        prompts = [
            build_generation_prompt(tokenizer, origin_prompt, args.system_prompt, args.disable_chat_template)
            for origin_prompt in origin_prompts
        ]

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        payload = {}
        for idx, (item, origin_prompt, output) in enumerate(zip(records, origin_prompts, outputs)):
            prediction = output.outputs[0].text.strip()
            payload[str(idx)] = {
                "origin_prompt": [
                    {
                        "role": "HUMAN",
                        "prompt": origin_prompt,
                    }
                ],
                "prediction": prediction,
                "refr": str(item.get("answer", "")).strip(),
            }

        task_output_path = output_dir / f"{task_id}.json"
        with task_output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved {task_output_path}")


if __name__ == "__main__":
    main()
