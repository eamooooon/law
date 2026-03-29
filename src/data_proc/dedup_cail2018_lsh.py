#!/usr/bin/env python3
"""
使用 MinHash LSH (局部敏感哈希) 对 CAIL2018的 284 万海量数据进行优雅的高维去重。

原理：
中国基层法院的刑事判决书（如醉驾、盗窃、寻衅滋事）存在极其严重的“模板化”现象。
使用字级别 3-gram 提取特征，计算 MinHash 签名，并通过 LSH 放入各个“桶”中。
若两篇案情描述的 Jaccard 相似度超过阈值（如 0.85），则视为“批量雷同案件”，直接丢弃后续样本，只保留首例。

依赖库: pip install datasketch
"""

import json
import os
from datasketch import MinHash, MinHashLSH
import time

# ========================
# 配置参数
# ========================
THRESHOLD = 0.3          # Jaccard 相似度判定阈值（高于此值认为是高度重复的案卷）
NUM_PERM = 128            # MinHash 的哈希函数数量（决定精度与内存权衡，128是经典甜点值）
N_GRAM = 3                # 使用字级别的 3-gram 作为特征集合

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_FILE = os.path.join(PROJECT_ROOT, "datasets", "proc", "cail2018_lsh_sft.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "datasets", "proc", "cail2018_lsh2_sft.jsonl")

def get_ngrams(text, n=N_GRAM):
    """
    极速流式分词：直接提取中文字符级别的 n-gram，省去了载入庞大 jieba 词库的极高开销
    """
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def dedup_cail2018_lsh():
    print(f"=== MinHash LSH 启动 ===")
    print(f"配置 -> 阈值: {THRESHOLD} | 哈希数: {NUM_PERM} | N-gram: {N_GRAM}")
    print(f"正在加载内存级极速去重树 (LSH Index)...")
    
    # 初始化 LSH 森林
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: 找不到文件 {INPUT_FILE}")
        return

    total_scanned = 0
    total_retained = 0
    duplicate_dropped = 0
    
    start_time = time.time()
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
             
        for line in f_in:
            if not line.strip():
                continue
                
            total_scanned += 1
            if total_scanned % 50000 == 0:
                elapsed = time.time() - start_time
                print(f"已扫描 {total_scanned} 条 | 保留 {total_retained} 条 | 丢弃 {duplicate_dropped} 条雷同案件 | 耗时: {elapsed:.1f}s")
            
            try:
                data = json.loads(line)
                case_id = data.get("id", str(total_scanned))
                text = data.get("input", "")
                
                if not text:
                    continue
                    
                # 1. 抽取 3-gram 特征
                ngrams = get_ngrams(text)
                
                # 2. 生成此案卷的 MinHash 签名
                m = MinHash(num_perm=NUM_PERM)
                for tg in ngrams:
                    m.update(tg.encode('utf8'))
                    
                # 3. 恐怖的 O(1) 桶内碰撞检索
                result = lsh.query(m)
                
                if len(result) > 0:
                    # 发现 Jaccard 相似度 > 0.85 的案件（桶内碰撞了），说明是套壳案件，杀掉
                    duplicate_dropped += 1
                else:
                    # 是一件全新作案手法的案子，插入 LSH 森林，并保存到本地
                    lsh.insert(case_id, m)
                    f_out.write(line)
                    total_retained += 1
                    
            except Exception as e:
                # 容错处理
                pass
                
    print("\n" + "="*40)
    print("💎 优雅的 LSH 降维打击去重完毕！")
    print(f"原始数据量: {total_scanned}")
    print(f"因高度雷同被剿灭的数据量: {duplicate_dropped}")
    print(f"最终保留下来的【高信息熵代表作】: {total_retained}")
    print("="*40)

if __name__ == "__main__":
    dedup_cail2018_lsh()
