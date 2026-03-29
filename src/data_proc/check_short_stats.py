#!/usr/bin/env python3
import json
import os
import glob

proc_dir = '/jiangdingfeng/zy/law/datasets/proc'
jsonl_files = glob.glob(os.path.join(proc_dir, '*.jsonl'))

print('=== Short Input Statistics ===', flush=True)
print(f'{"Filename":<30} | {"Total":<10} | {"< 100 chars":<15} | {"< 200 chars":<15}', flush=True)
print('-' * 80, flush=True)

for file_path in sorted(jsonl_files):
    if 'sampled' in file_path: continue
    file_name = os.path.basename(file_path)
    if 'cail2018' in file_name: continue  # Skip the 2.8M file to save time, its input is just a fact string anyway.
    
    total = 0
    under_100 = 0
    under_200 = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                input_text = data.get('input', '')
                input_len = len(input_text)
                total += 1
                if input_len < 100: under_100 += 1
                if input_len < 200: under_200 += 1
            except Exception as e:
                pass
                
    if total > 0:
        p100 = (under_100 / total) * 100
        p200 = (under_200 / total) * 100
        print(f'{file_name:<30} | {total:<10} | {under_100:<5} ({p100:5.1f}%) | {under_200:<5} ({p200:5.1f}%)', flush=True)

print('-' * 80, flush=True)
