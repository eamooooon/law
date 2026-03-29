#!/usr/bin/env python3
"""
遍历 datasets/proc 目录下的所有 SFT 文件，
过滤掉 input 字段字符数少于 100 的弱信息数据。

原因：由于我们准备使用这些数据做大模型思维链 (CoT) 的反推生成基座，
如果原文长度过短（如仅一句话），大模型无法从中提炼出有意义的推理过程，会导致生成的 reasoning 发生强行注水或幻觉。
"""

import json
import os
import glob

# 配置
MIN_CHARS = 100
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC_DIR = os.path.join(PROJECT_ROOT, "datasets", "proc")

def filter_short_inputs():
    print(f"=== SFT Dataset Quality Refinement ===")
    print(f"Target: Remove any records where strictly 'input' length < {MIN_CHARS} chars.")
    
    jsonl_files = glob.glob(os.path.join(PROC_DIR, "*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {PROC_DIR}")
        return

    total_scanned = 0
    total_removed = 0
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        # 生成一个临时文件来写，成功后再替换原文件
        temp_file_path = file_path + ".tmp"
        
        file_total = 0
        file_removed = 0
        
        with open(file_path, "r", encoding="utf-8") as f_in, \
             open(temp_file_path, "w", encoding="utf-8") as f_out:
                 
            for line in f_in:
                if not line.strip():
                    continue
                file_total += 1
                
                try:
                    data = json.loads(line)
                    input_text = data.get("input", "")
                    
                    if len(input_text) < MIN_CHARS:
                        file_removed += 1
                        continue # 丢弃短于限制的数据
                        
                    f_out.write(line)
                except json.JSONDecodeError:
                    pass
                    
        total_scanned += file_total
        total_removed += file_removed
        
        # 覆盖原文件
        os.replace(temp_file_path, file_path)
        
        if file_total > 0:
            removed_pct = (file_removed / file_total) * 100
            print(f"Processed \033[92m{file_name}\033[0m: Removed {file_removed} / {file_total} ({removed_pct:.1f}%) too short inputs.")

    print("-" * 50)
    print(f"Global Refinement Complete!")
    print(f"Total Records Scanned: {total_scanned}")
    print(f"\033[91mTotal Records Deleted (< {MIN_CHARS} chars)\033[0m: {total_removed}")
    print(f"\033[92mTotal Records Retained\033[0m: {total_scanned - total_removed}")
    
if __name__ == "__main__":
    filter_short_inputs()
