import argparse
import datetime
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TASK_SUITES = {
    "plan_b": [
        "abercrombie",
        "canada_tax_court_outcomes",
        "overruling",
        "function_of_decision_section",
        "contract_nli_confidentiality_of_agreement",
        "contract_nli_explicit_identification",
        "contract_nli_inclusion_of_verbally_conveyed_information",
        "contract_nli_limited_use",
        "contract_nli_no_licensing",
        "contract_nli_notice_on_compelled_disclosure",
        "contract_nli_return_of_confidential_information",
        "contract_nli_survival_of_obligations",
        "consumer_contracts_qa",
        "hearsay",
        "personal_jurisdiction",
        "telemarketing_sales_rule",
        "legal_reasoning_causality",
        "successor_liability",
        "rule_qa",
        "sara_entailment",
        "scalr",
        "definition_classification",
        "definition_extraction",
        "textualism_tool_plain",
        "textualism_tool_dictionaries",
        "jcrew_blocker",
        "proa",
        "nys_judicial_ethics",
        "supply_chain_disclosure_best_practice_verification",
        "supply_chain_disclosure_disclosed_verification",
        "contract_nli_permissible_copy",
        "contract_nli_sharing_with_employees",
        "maud_knowledge_definition",
        "maud_specific_performance",
        "maud_type_of_consideration",
        "cuad_governing_law",
        "cuad_change_of_control",
        "cuad_source_code_escrow",
        "opp115_policy_change",
        "opp115_do_not_track",
        "learned_hands_courts",
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic LegalBench evaluator")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/sft-qwen2.5-3b-lr2e5-all-2",
        help="Model directory for vLLM and tokenizer loading.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/eval/legalbench",
        help="LegalBench dataset path.",
    )
    parser.add_argument(
        "--task_limit",
        type=int,
        default=10,
        help="Evaluate the first N tasks. Use 0 or negative for all tasks.",
    )
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="Comma-separated task names to run. Overrides --task_limit when set.",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="",
        choices=sorted(TASK_SUITES.keys()),
        help="Named task suite to run. Used when --task_names is not set.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default is deterministic decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top-p sampling parameter.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a highly accurate legal assistant. Return only the final answer label or span, with no explanation unless the task requires a free-form answer.",
        help="System prompt used for evaluation.",
    )
    parser.add_argument(
        "--force_cot",
        action="store_true",
        help="If set, explicitly asks the model to think step by step.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="results",
        help="Root directory to save reports and raw responses.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory. If set, results are written here directly.",
    )
    parser.add_argument(
        "--max_contract_chars",
        type=int,
        default=3000,
        help="Maximum contract/context characters included in prompt.",
    )
    return parser.parse_args()


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def infer_answer_labels(dataset) -> List[str]:
    answers = []
    seen = set()
    for value in dataset["answer"]:
        normalized = normalize_label(str(value))
        if normalized not in seen:
            seen.add(normalized)
            answers.append(str(value).strip())

    if len(answers) <= 12 and all(len(answer) <= 80 for answer in answers):
        return answers
    return []


def label_instruction(answer_labels: List[str]) -> str:
    if not answer_labels:
        return "Reply with only the final answer."

    normalized = {normalize_label(label) for label in answer_labels}
    if normalized <= {"yes", "no"}:
        return "Reply with only Yes or No."
    if normalized <= {"a", "b", "c", "d", "e"}:
        return "Reply with only the best answer letter."

    label_text = ", ".join(answer_labels)
    return f"Reply with only one of: {label_text}."


def build_prompt(
    item: Dict,
    cols: List[str],
    tokenizer,
    system_prompt: str,
    max_contract_chars: int,
    answer_labels: List[str],
) -> str:
    answer_instruction = label_instruction(answer_labels)
    statute_context_chars = max(max_contract_chars, 12000)

    if "choice_0" in cols:
        choices = [item[f"choice_{i}"] for i in range(5) if f"choice_{i}" in cols]
        options_text = "\n".join(f"({chr(97 + i)}) {choice}" for i, choice in enumerate(choices))
        user_content = (
            f"{item['question']}\n\n"
            f"{options_text}\n\n"
            "Reply with only the best answer choice using either the letter a-e or the index 0-4."
        )
    elif "question" in cols and "contract" in cols:
        user_content = (
            f"Contract:\n{item['contract'][:max_contract_chars]}\n\n"
            f"Question: {item['question']}\n\n"
            f"{answer_instruction}"
        )
    elif "question" in cols and "text" in cols:
        user_content = (
            f"Text:\n{item['text'][:max_contract_chars]}\n\n"
            f"Question: {item['question']}\n\n"
            f"{answer_instruction}"
        )
    elif "policy" in cols and "claim" in cols:
        user_content = (
            f"Policy:\n{item['policy'][:max_contract_chars]}\n\n"
            f"Claim:\n{item['claim']}\n\n"
            f"{answer_instruction}"
        )
    elif "contract" in cols:
        user_content = (
            f"Contract Scenario:\n{item['contract'][:max_contract_chars]}\n\n"
            f"{answer_instruction}"
        )
    elif "text" in cols and "citation" in cols:
        user_content = (
            f"Legal Statement:\n{item['text']}\n\n"
            f"Citation: {item['citation']}\n\n"
            f"{answer_instruction}"
        )
    elif "Citation" in cols and "Paragraph" in cols:
        user_content = (
            f"Citation: {item['Citation']}\n\n"
            f"Paragraph:\n{item['Paragraph']}\n\n"
            f"{answer_instruction}"
        )
    elif "question" in cols and "year" in cols:
        user_content = (
            f"Year: {item['year']}\n\n"
            f"Question: {item['question']}\n\n"
            f"{answer_instruction}"
        )
    elif "question" in cols:
        user_content = f"{item['question']}\n\n{answer_instruction}"
    elif "issue" in cols and "text" in cols:
        user_content = (
            f"Issue: {item['issue']}\n\n"
            f"Fact Pattern:\n{item['text']}\n\n"
            f"{answer_instruction}"
        )
    elif "slice" in cols:
        user_content = (
            f"Context: {item.get('slice', '')}\n"
            f"Text: {item['text']}\n\n"
            f"{answer_instruction}"
        )
    elif "description" in cols and "statute" in cols:
        statute_instruction = answer_instruction
        if "how much tax" in str(item.get("question", "")).lower():
            statute_instruction = "Compute the exact final dollar amount. Reply with only a dollar amount like $12345."
        user_content = (
            f"Statute: {item['statute'][:statute_context_chars]}\n"
            f"Description: {item['description']}\n"
            f"Question: {item.get('question', '')}\n\n"
            f"Text: {item['text'][:statute_context_chars]}\n\n"
            f"{statute_instruction}"
        )
    elif "text" in cols and "description" in cols:
        user_content = (
            f"Policy Text:\n{item['text'][:max_contract_chars]}\n\n"
            f"Description: {item['description']}\n\n"
            f"{answer_instruction}"
        )
    else:
        user_content = f"{item.get('text', '')}\n\n{answer_instruction}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    text = strip_think_blocks(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def find_matching_label(prediction: str, answer_labels: List[str]) -> Optional[str]:
    if not answer_labels:
        return None

    normalized_prediction = normalize_text(prediction)
    normalized_labels = [(label, normalize_label(label)) for label in answer_labels]

    for label, normalized_label_text in sorted(normalized_labels, key=lambda item: len(item[1]), reverse=True):
        if not normalized_label_text:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_label_text)}(?![a-z0-9])"
        if re.search(pattern, normalized_prediction):
            return label

    return None


def normalize_choice_answer(text: str) -> str:
    normalized = normalize_text(text)
    digit_to_letter = {"0": "a", "1": "b", "2": "c", "3": "d", "4": "e"}
    if normalized in digit_to_letter:
        return digit_to_letter[normalized]
    return normalized


def extract_final_answer(prediction: str, truth: str, cols: List[str], answer_labels: List[str]) -> str:
    normalized_prediction = normalize_text(prediction)
    normalized_truth = normalize_text(truth)

    if normalized_truth in {"yes", "no"}:
        match = re.search(r"\b(yes|no)\b", normalized_prediction)
        if match:
            return match.group(1)

    if "choice_0" in cols:
        digit_match = re.search(r"\b([0-4])\b", normalized_prediction)
        if digit_match:
            return digit_match.group(1)
        match = re.search(r"\b([a-e])\b", normalized_prediction)
        if match:
            return match.group(1)

    matched_label = find_matching_label(prediction, answer_labels)
    if matched_label is not None:
        return matched_label

    lines = [line.strip() for line in strip_think_blocks(prediction).splitlines() if line.strip()]
    if not lines:
        return normalized_prediction

    last_line = normalize_text(lines[-1])
    if last_line.startswith("final answer:"):
        last_line = last_line[len("final answer:"):].strip()
    if last_line.startswith("answer:"):
        last_line = last_line[len("answer:"):].strip()
    matched_label = find_matching_label(last_line, answer_labels)
    if matched_label is not None:
        return matched_label
    return last_line


def is_correct(prediction: str, truth: str, cols: List[str], answer_labels: List[str]) -> Tuple[bool, str]:
    normalized_truth = normalize_text(truth)
    extracted = extract_final_answer(prediction, truth, cols, answer_labels)
    extracted = normalize_text(extracted)

    if "choice_0" in cols:
        normalized_truth = normalize_choice_answer(truth)
        extracted = normalize_choice_answer(extracted)

    if not extracted:
        return False, extracted
    if extracted == normalized_truth:
        return True, extracted
    if normalized_truth in {"yes", "no"} and extracted in {"yes", "no"}:
        return extracted == normalized_truth, extracted
    if "choice_0" in cols and normalized_truth in {"a", "b", "c", "d", "e"}:
        return extracted == normalized_truth, extracted
    return normalized_truth in extracted or extracted in normalized_truth, extracted


def main():
    args = parse_args()
    model_name = os.path.basename(args.model_dir.rstrip("/"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = (
        args.output_dir
        if args.output_dir.strip()
        else os.path.join(args.result_root, f"{model_name}_legalbench2_{timestamp}")
    )
    responses_dir = os.path.join(result_dir, "responses")
    output_json = os.path.join(result_dir, "evaluation_results.json")
    output_md = os.path.join(result_dir, "evaluation_report.md")

    os.makedirs(responses_dir, exist_ok=True)

    print("=========================================")
    print("LegalBench deterministic evaluation")
    print("=========================================")
    print(f"Model: {args.model_dir}")
    print(f"Dataset: {args.dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    all_tasks = get_dataset_config_names(args.dataset_path, trust_remote_code=True)
    if args.task_names.strip():
        selected = {name.strip() for name in args.task_names.split(",") if name.strip()}
        all_tasks = [task for task in all_tasks if task in selected]
    elif args.task_suite:
        selected = set(TASK_SUITES[args.task_suite])
        all_tasks = [task for task in all_tasks if task in selected]
    elif args.task_limit and args.task_limit > 0:
        all_tasks = all_tasks[:args.task_limit]

    system_prompt = args.system_prompt
    if args.force_cot:
        system_prompt = f"{system_prompt} Think step by step before the final answer."

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    start_time = time.time()
    total_questions_all = 0
    total_correct_all = 0
    results_log = []

    for task_idx, task_name in enumerate(all_tasks, 1):
        print(f"\n[{task_idx}/{len(all_tasks)}] {task_name}")
        dataset = load_dataset(args.dataset_path, task_name, split="test", trust_remote_code=True)
        if len(dataset) == 0:
            print("Skip empty task.")
            continue

        cols = dataset.column_names
        answer_labels = infer_answer_labels(dataset)
        prompts = [
            build_prompt(item, cols, tokenizer, system_prompt, args.max_contract_chars, answer_labels)
            for item in dataset
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        correct = 0
        task_records = []
        for idx, output in enumerate(outputs):
            raw_prediction = output.outputs[0].text.strip()
            truth = str(dataset[idx]["answer"]).strip()
            matched, extracted = is_correct(raw_prediction, truth, cols, answer_labels)
            if matched:
                correct += 1

            task_records.append(
                {
                    "index": idx,
                    "truth": truth,
                    "prediction_raw": raw_prediction,
                    "prediction_extracted": extracted,
                    "matched": matched,
                }
            )

        task_acc = correct / len(dataset)
        total_questions_all += len(dataset)
        total_correct_all += correct
        print(f"Accuracy: {task_acc * 100:.2f}% ({correct}/{len(dataset)})")

        results_log.append(
            {
                "Task Name": task_name,
                "Total Samples": len(dataset),
                "Correct": correct,
                "Accuracy (%)": round(task_acc * 100, 2),
            }
        )

        task_path = os.path.join(responses_dir, f"{task_name}.jsonl")
        with open(task_path, "w", encoding="utf-8") as f:
            for record in task_records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    elapsed_minutes = (time.time() - start_time) / 60
    micro_acc = (total_correct_all / total_questions_all) * 100 if total_questions_all else 0
    macro_acc = (
        sum(item["Accuracy (%)"] for item in results_log) / len(results_log)
        if results_log
        else 0
    )

    summary = {
        "model_dir": args.model_dir,
        "dataset_path": args.dataset_path,
        "task_count": len(results_log),
        "total_questions": total_questions_all,
        "total_correct": total_correct_all,
        "micro_accuracy_percent": round(micro_acc, 4),
        "macro_accuracy_percent": round(macro_acc, 4),
        "elapsed_minutes": round(elapsed_minutes, 4),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "force_cot": args.force_cot,
        "system_prompt": system_prompt,
        "task_suite": args.task_suite,
    }
    payload = {"summary": summary, "tasks": results_log}

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# LegalBench Evaluation Report",
        "",
        f"- Model: {args.model_dir}",
        f"- Dataset: {args.dataset_path}",
        f"- Generated At: {summary['generated_at']}",
        f"- Elapsed Minutes: {summary['elapsed_minutes']:.2f}",
        f"- Temperature: {args.temperature}",
        f"- Top-p: {args.top_p}",
        f"- Max Tokens: {args.max_tokens}",
        f"- Force CoT: {args.force_cot}",
        "",
        "## Summary",
        "",
        f"- Task Count: {summary['task_count']}",
        f"- Total Questions: {summary['total_questions']}",
        f"- Total Correct: {summary['total_correct']}",
        f"- Micro Accuracy (%): {summary['micro_accuracy_percent']:.2f}",
        f"- Macro Accuracy (%): {summary['macro_accuracy_percent']:.2f}",
        "",
        "## Task Details",
        "",
        "| Task Name | Total Samples | Correct | Accuracy (%) |",
        "|---|---:|---:|---:|",
    ]
    for item in results_log:
        md_lines.append(
            f"| {item['Task Name']} | {item['Total Samples']} | {item['Correct']} | {item['Accuracy (%)']:.2f} |"
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n=========================================")
    print("Evaluation complete")
    print("=========================================")
    print(f"Micro Accuracy: {micro_acc:.2f}%")
    print(f"Macro Accuracy: {macro_acc:.2f}%")
    print(f"JSON: {os.path.abspath(output_json)}")
    print(f"Markdown: {os.path.abspath(output_md)}")
    print(f"Responses: {os.path.abspath(responses_dir)}")


if __name__ == "__main__":
    main()
