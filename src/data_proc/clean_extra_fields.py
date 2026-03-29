#!/usr/bin/env python3
"""
清洗数据集中的多余字段
移除 _source_file, _source_line, _sampled 三个字段
"""
import json
import sys
from pathlib import Path

def clean_jsonl_file(input_path, output_path=None):
    """
    清洗 JSONL 文件，移除指定字段
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径，默认覆盖原文件
    """
    if output_path is None:
        output_path = input_path
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"错误：文件不存在 {input_path}")
        return False
    
    # 需要移除的字段
    fields_to_remove = {'_source_file', '_source_line', '_sampled'}
    
    cleaned_count = 0
    total_count = 0
    
    try:
        # 先读取所有行并处理
        cleaned_lines = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                total_count += 1
                try:
                    data = json.loads(line)
                    
                    # 移除多余字段
                    original_keys = set(data.keys())
                    for field in fields_to_remove:
                        data.pop(field, None)
                    
                    # 检查是否真的有字段被移除
                    if len(original_keys) > len(data):
                        cleaned_count += 1
                    
                    cleaned_lines.append(json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError as e:
                    print(f"警告：第 {total_count} 行 JSON 解析失败: {e}")
                    cleaned_lines.append(line.rstrip('\n'))
        
        # 写回文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')
        
        print(f"✓ 清洗完成: {input_path}")
        print(f"  总行数: {total_count}")
        print(f"  清洗行数: {cleaned_count}")
        return True
        
    except Exception as e:
        print(f"错误：处理失败 - {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python clean_extra_fields.py <文件1> [文件2] ...")
        sys.exit(1)
    
    files = sys.argv[1:]
    for file_path in files:
        clean_jsonl_file(file_path)
