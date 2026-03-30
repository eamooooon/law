import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LawBench CSV metrics into JSON and Markdown reports.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing lawbench_*.csv files.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/eval/LawBench/data",
        help="LawBench data root containing zero_shot and one_shot subfolders.",
    )
    parser.add_argument("--model_dir", type=str, default="", help="Optional model path for the report header.")
    return parser.parse_args()


def load_sample_count(data_root: Path, split: str, task_id: str) -> Optional[int]:
    task_path = data_root / split / f"{task_id}.json"
    if not task_path.exists():
        return None
    with task_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return len(records)


def round_percent(value: float) -> float:
    return round(value * 100.0, 4)


def read_split_metrics(results_dir: Path, data_root: Path, split: str) -> Optional[Dict]:
    csv_path = results_dir / f"lawbench_{split}.csv"
    if not csv_path.exists():
        return None

    tasks: List[Dict] = []
    total_weight = 0
    weighted_score_sum = 0.0
    weighted_abstention_sum = 0.0
    score_sum = 0.0
    abstention_sum = 0.0
    model_name = ""
    group_buckets: Dict[str, List[Dict]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row["task"].strip()
            model_name = model_name or row.get("model_name", "").strip()
            score = float(row["score"])
            abstention_rate = float(row["abstention_rate"])
            sample_count = load_sample_count(data_root, split, task_id)

            score_sum += score
            abstention_sum += abstention_rate

            if sample_count is not None:
                total_weight += sample_count
                weighted_score_sum += score * sample_count
                weighted_abstention_sum += abstention_rate * sample_count

            task_info = {
                "task_id": task_id,
                "sample_count": sample_count,
                "score_percent": round_percent(score),
                "abstention_rate_percent": round_percent(abstention_rate),
            }
            tasks.append(task_info)
            group_buckets[task_id.split("-", 1)[0]].append(task_info)

    if not tasks:
        return None

    task_count = len(tasks)
    macro_score = score_sum / task_count
    macro_abstention = abstention_sum / task_count
    weighted_score = (weighted_score_sum / total_weight) if total_weight else macro_score
    weighted_abstention = (weighted_abstention_sum / total_weight) if total_weight else macro_abstention

    groups = []
    for group_name in sorted(group_buckets):
        group_tasks = group_buckets[group_name]
        group_sample_count = sum(task["sample_count"] or 0 for task in group_tasks)
        group_macro_score = sum(task["score_percent"] for task in group_tasks) / len(group_tasks)
        group_macro_abstention = sum(task["abstention_rate_percent"] for task in group_tasks) / len(group_tasks)
        groups.append(
            {
                "group": group_name,
                "task_count": len(group_tasks),
                "total_samples": group_sample_count if group_sample_count > 0 else None,
                "macro_score_percent": round(group_macro_score, 4),
                "macro_abstention_rate_percent": round(group_macro_abstention, 4),
            }
        )

    return {
        "split": split,
        "model_name": model_name,
        "task_count": task_count,
        "total_samples": total_weight if total_weight > 0 else None,
        "macro_score_percent": round_percent(macro_score),
        "weighted_score_percent": round_percent(weighted_score),
        "macro_abstention_rate_percent": round_percent(macro_abstention),
        "weighted_abstention_rate_percent": round_percent(weighted_abstention),
        "groups": groups,
        "tasks": sorted(tasks, key=lambda item: item["task_id"]),
    }


def build_overall_summary(split_summaries: List[Dict]) -> Dict:
    task_count = sum(item["task_count"] for item in split_summaries)
    macro_score = sum(item["macro_score_percent"] for item in split_summaries) / len(split_summaries)
    macro_abstention = sum(item["macro_abstention_rate_percent"] for item in split_summaries) / len(split_summaries)

    weighted_scores = []
    weighted_abstentions = []
    total_samples = 0
    for item in split_summaries:
        samples = item["total_samples"] or 0
        if samples > 0:
            total_samples += samples
            weighted_scores.append((item["weighted_score_percent"], samples))
            weighted_abstentions.append((item["weighted_abstention_rate_percent"], samples))

    if total_samples > 0:
        weighted_score = round(sum(score * samples for score, samples in weighted_scores) / total_samples, 4)
        weighted_abstention = round(
            sum(rate * samples for rate, samples in weighted_abstentions) / total_samples, 4
        )
        total_samples_value: Optional[int] = total_samples
    else:
        weighted_score = round(macro_score, 4)
        weighted_abstention = round(macro_abstention, 4)
        total_samples_value = None

    return {
        "split_count": len(split_summaries),
        "task_count": task_count,
        "total_samples": total_samples_value,
        "macro_score_percent": round(macro_score, 4),
        "weighted_score_percent": weighted_score,
        "macro_abstention_rate_percent": round(macro_abstention, 4),
        "weighted_abstention_rate_percent": weighted_abstention,
    }


def write_json_report(results_dir: Path, payload: Dict) -> None:
    path = results_dir / "evaluation_results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown_report(results_dir: Path, payload: Dict) -> None:
    path = results_dir / "evaluation_report.md"
    lines = [
        "# LawBench Evaluation Report",
        "",
        f"- Model: {payload['summary']['model_dir']}",
        f"- Data Root: {payload['summary']['data_root']}",
        f"- Generated At: {payload['summary']['generated_at']}",
        "",
        "## Overall Summary",
        "",
        f"- Split Count: {payload['overall']['split_count']}",
        f"- Task Count: {payload['overall']['task_count']}",
        f"- Total Samples: {payload['overall']['total_samples'] if payload['overall']['total_samples'] is not None else 'N/A'}",
        f"- Macro Score (%): {payload['overall']['macro_score_percent']:.4f}",
        f"- Weighted Score (%): {payload['overall']['weighted_score_percent']:.4f}",
        f"- Macro Abstention Rate (%): {payload['overall']['macro_abstention_rate_percent']:.4f}",
        f"- Weighted Abstention Rate (%): {payload['overall']['weighted_abstention_rate_percent']:.4f}",
        "",
        "## Split Summary",
        "",
        "| Split | Task Count | Total Samples | Macro Score (%) | Weighted Score (%) | Macro Abstention (%) | Weighted Abstention (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for split_result in payload["splits"]:
        total_samples = split_result["total_samples"] if split_result["total_samples"] is not None else "N/A"
        lines.append(
            f"| {split_result['split']} | {split_result['task_count']} | {total_samples} | "
            f"{split_result['macro_score_percent']:.4f} | {split_result['weighted_score_percent']:.4f} | "
            f"{split_result['macro_abstention_rate_percent']:.4f} | {split_result['weighted_abstention_rate_percent']:.4f} |"
        )

    for split_result in payload["splits"]:
        lines.extend(
            [
                "",
                f"## {split_result['split']}",
                "",
                "### Group Summary",
                "",
                "| Group | Task Count | Total Samples | Macro Score (%) | Macro Abstention (%) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for group in split_result["groups"]:
            total_samples = group["total_samples"] if group["total_samples"] is not None else "N/A"
            lines.append(
                f"| {group['group']} | {group['task_count']} | {total_samples} | "
                f"{group['macro_score_percent']:.4f} | {group['macro_abstention_rate_percent']:.4f} |"
            )

        lines.extend(
            [
                "",
                "### Task Details",
                "",
                "| Task ID | Samples | Score (%) | Abstention (%) |",
                "|---|---:|---:|---:|",
            ]
        )
        for task in split_result["tasks"]:
            sample_count = task["sample_count"] if task["sample_count"] is not None else "N/A"
            lines.append(
                f"| {task['task_id']} | {sample_count} | {task['score_percent']:.4f} | "
                f"{task['abstention_rate_percent']:.4f} |"
            )

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    data_root = Path(args.data_root)
    split_summaries = []
    for split in ("zero_shot", "one_shot"):
        split_result = read_split_metrics(results_dir, data_root, split)
        if split_result is not None:
            split_summaries.append(split_result)

    if not split_summaries:
        raise FileNotFoundError(f"No lawbench_*.csv files found under {results_dir}")

    payload = {
        "summary": {
            "model_dir": args.model_dir or split_summaries[0].get("model_name", ""),
            "data_root": str(data_root),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        "overall": build_overall_summary(split_summaries),
        "splits": split_summaries,
    }

    write_json_report(results_dir, payload)
    write_markdown_report(results_dir, payload)
    print(f"Saved {results_dir / 'evaluation_results.json'}")
    print(f"Saved {results_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
