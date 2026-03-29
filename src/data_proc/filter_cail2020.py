"""
过滤 cail2020_ydlj_sft.jsonl 中无效数据：
  - reasoning 字段为空字符串
  - output 字段为 "unknown"（忽略大小写）
结果写入同目录 xxx_cleaned.jsonl
"""

import json
import os

SRC = "/jiangdingfeng/zy/law/datasets/proc/cail2020_ydlj_sft.jsonl"
DST = SRC.replace(".jsonl", "_cleaned.jsonl")

total = kept = 0
bad_reasoning = bad_output = 0

with open(SRC, "r", encoding="utf-8") as fin, \
     open(DST, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        obj = json.loads(line)

        # 跳过 reasoning 为空的情况
        if obj.get("reasoning", "") == "":
            bad_reasoning += 1
            continue

        # 跳过 output 为 unknown 的情况
        if str(obj.get("output", "")).strip().lower() == "unknown":
            bad_output += 1
            continue

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

removed = total - kept
print(f"总行数    : {total}")
print(f"保留行数  : {kept}")
print(f"剔除行数  : {removed}  (reasoning空={bad_reasoning}, output=unknown={bad_output})")
print(f"输出文件  : {DST}")
