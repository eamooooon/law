#!/usr/bin/env python3
"""
Diverse sampling with lightweight embedding + KMeans centroid selection.

Default workflow:
1) Read two source JSONL files from datasets/proc.
2) Reservoir-sample up to 50,000 records as candidate pool.
3) Vectorize texts with lightweight TF-IDF char n-gram embedding.
4) Run MiniBatchKMeans with K=5000 (or adjusted if pool is smaller).
5) Select one nearest sample to each cluster centroid.
6) Save selected and remaining records as new JSON files in the same target directory.
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_INPUT_FILES = [
    # os.path.join(PROJECT_ROOT, "datasets", "proc", "cail2018_lsh2_sft.jsonl"),
    # os.path.join(PROJECT_ROOT, "datasets", "proc", "cail2019_cjrc_sft.jsonl"),
    os.path.join(PROJECT_ROOT, "datasets", "proc", "jec_qa_sft.jsonl"),
]
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "proc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diverse sample selection by embedding + KMeans.")
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="Input JSONL files. Default is cail2018_lsh2_sft.jsonl + cail2019_cjrc_sft.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for selected and remaining JSON files.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100000,
        help="Candidate pool size before clustering. Use 0 to keep all records in memory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Target selected sample size (K in KMeans).",
    )
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=["instruction", "input", "output"],
        help="Fields used to build embedding text.",
    )
    parser.add_argument("--max-features", type=int, default=50000, help="TF-IDF max features.")
    parser.add_argument("--min-df", type=int, default=2, help="TF-IDF min_df.")
    parser.add_argument("--max-df", type=float, default=0.98, help="TF-IDF max_df.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=50000, help="Log progress every N rows/clusters.")
    parser.add_argument("--kmeans-verbose", type=int, default=0, help="MiniBatchKMeans verbose level.")
    return parser.parse_args()


def load_pool_reservoir(input_files: Sequence[str], pool_size: int, seed: int, log_every: int) -> List[Dict[str, Any]]:
    t0 = time.time()
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    seen = 0

    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        file_seen = 0
        print(f"[INFO] reading file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict):
                    continue

                record["_source_file"] = os.path.basename(file_path)
                record["_source_line"] = line_no

                seen += 1
                file_seen += 1
                if pool_size <= 0:
                    pool.append(record)
                    continue

                if len(pool) < pool_size:
                    pool.append(record)
                else:
                    j = rng.randint(1, seen)
                    if j <= pool_size:
                        pool[j - 1] = record

                if log_every > 0 and seen % log_every == 0:
                    elapsed = time.time() - t0
                    speed = seen / max(elapsed, 1e-6)
                    print(
                        f"[PROGRESS] loading seen={seen} pool={len(pool)} elapsed={elapsed:.1f}s speed={speed:.1f} rows/s"
                    )

        print(f"[INFO] finished file: {file_path} valid_rows={file_seen}")

    if not pool:
        raise ValueError("No valid records loaded from inputs.")

    print(f"[INFO] total seen records: {seen}")
    print(f"[INFO] candidate pool size: {len(pool)}")
    print(f"[INFO] loading elapsed: {time.time() - t0:.1f}s")
    return pool


def build_text(record: Dict[str, Any], text_fields: Sequence[str]) -> str:
    parts: List[str] = []
    for field in text_fields:
        value = record.get(field, "")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def vectorize(records: Sequence[Dict[str, Any]], text_fields: Sequence[str], max_features: int, min_df: int, max_df: float):
    t0 = time.time()
    print("[INFO] vectorizing corpus...")
    corpus = [build_text(r, text_fields) for r in records]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    x = vectorizer.fit_transform(corpus)
    print(f"[INFO] vectorized shape: {x.shape}")
    print(f"[INFO] vectorization elapsed: {time.time() - t0:.1f}s")
    return x


def pick_centroid_nearest(x, labels: np.ndarray, centers: np.ndarray, log_every: int) -> List[int]:
    t0 = time.time()
    print("[INFO] selecting centroid-nearest samples...")
    selected: List[int] = []
    n_clusters = centers.shape[0]

    for cluster_id in range(n_clusters):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) == 0:
            continue

        dists = pairwise_distances(x[idxs], centers[cluster_id].reshape(1, -1), metric="euclidean").ravel()
        best_local = int(np.argmin(dists))
        selected.append(int(idxs[best_local]))

        if log_every > 0 and (cluster_id + 1) % log_every == 0:
            elapsed = time.time() - t0
            speed = (cluster_id + 1) / max(elapsed, 1e-6)
            print(
                f"[PROGRESS] centroid_pick cluster={cluster_id + 1}/{n_clusters} elapsed={elapsed:.1f}s speed={speed:.1f} clusters/s"
            )

    print(f"[INFO] centroid selection elapsed: {time.time() - t0:.1f}s")
    return selected


def ensure_exact_size(selected: List[int], total_n: int, target_n: int, seed: int) -> List[int]:
    if len(selected) == target_n:
        return selected

    selected_set = set(selected)
    if len(selected) > target_n:
        return selected[:target_n]

    rng = random.Random(seed)
    remaining = [i for i in range(total_n) if i not in selected_set]
    need = target_n - len(selected)
    if need > len(remaining):
        need = len(remaining)
    selected.extend(rng.sample(remaining, k=need))
    return selected


def write_json(path: str, records: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(list(records), fout, ensure_ascii=False, indent=2)


def main() -> None:
    all_start = time.time()
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    pool = load_pool_reservoir(args.input_files, args.pool_size, args.seed, args.log_every)

    if len(pool) < 2:
        raise ValueError("Not enough records for clustering.")

    target_k = min(args.sample_size, len(pool))
    if target_k < args.sample_size:
        print(f"[WARN] sample-size={args.sample_size} > pool-size={len(pool)}; use K={target_k}")

    x = vectorize(
        pool,
        text_fields=args.text_fields,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    print(f"[INFO] running MiniBatchKMeans: n_clusters={target_k}, max_iter=100, n_init=10")
    kmeans_start = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=target_k,
        random_state=args.seed,
        batch_size=2048,
        n_init=10,
        max_iter=100,
        verbose=args.kmeans_verbose,
    )
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_
    print(f"[INFO] kmeans elapsed: {time.time() - kmeans_start:.1f}s")

    selected_idx = pick_centroid_nearest(x, labels, centers, args.log_every)
    selected_idx = ensure_exact_size(selected_idx, total_n=len(pool), target_n=target_k, seed=args.seed)
    selected_idx_set = set(selected_idx)

    selected_records = [pool[i] for i in selected_idx]
    remaining_records = [pool[i] for i in range(len(pool)) if i not in selected_idx_set]

    for record in selected_records:
        record["_sampled"] = "selected"
    for record in remaining_records:
        record["_sampled"] = "remaining"

    selected_name = f"selected_{target_k}.json"
    remaining_name = f"remaining_{len(remaining_records)}.json"
    selected_path = os.path.join(args.output_dir, selected_name)
    remaining_path = os.path.join(args.output_dir, remaining_name)

    write_start = time.time()
    print("[INFO] writing output files...")
    write_json(selected_path, selected_records)
    write_json(remaining_path, remaining_records)
    print(f"[INFO] writing elapsed: {time.time() - write_start:.1f}s")

    print("=" * 70)
    print("Done")
    print(f"Selected file : {selected_path}")
    print(f"Remaining file: {remaining_path}")
    print(f"Pool size     : {len(pool)}")
    print(f"Selected size : {len(selected_records)}")
    print(f"Remaining size: {len(remaining_records)}")
    print(f"Total elapsed : {time.time() - all_start:.1f}s")


if __name__ == "__main__":
    main()
