#!/usr/bin/env python3
"""
过滤 output 和 output_model 一致的记录
"""
import json
from pathlib import Path


def filter_consistent_records(input_file: str, output_file: str):
    """
    过滤出 output 和 output_model 字段一致的记录

    Args:
        input_file: 输入的 jsonl 文件路径
        output_file: 输出的 jsonl 文件路径
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return

    total_count = 0
    consistent_count = 0
    inconsistent_count = 0
    inconsistent_examples = []

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            total_count += 1

            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                inconsistent_count += 1
                continue

            output = data.get('output', '').strip()
            output_model = data.get('output_model', '').strip()

            if output == output_model:
                # 一致,保留
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                consistent_count += 1
            else:
                # 不一致,记录示例
                inconsistent_count += 1
                if len(inconsistent_examples) < 5:  # 只保存前5个示例
                    inconsistent_examples.append({
                        'id': data.get('id', 'unknown'),
                        'output': output,
                        'output_model': output_model
                    })

    # 打印统计信息
    print(f"\n处理完成!")
    print(f"总记录数: {total_count}")
    print(f"一致记录数: {consistent_count} ({consistent_count/total_count*100:.2f}%)")
    print(f"不一致记录数: {inconsistent_count} ({inconsistent_count/total_count*100:.2f}%)")
    print(f"\n输出文件: {output_file}")

    # 显示不一致的示例
    if inconsistent_examples:
        print(f"\n不一致示例 (前5个):")
        for i, example in enumerate(inconsistent_examples, 1):
            print(f"\n示例 {i} (ID: {example['id']}):")
            print(f"  output:      {example['output']}")
            print(f"  output_model: {example['output_model']}")


if __name__ == '__main__':
    input_file = 'datasets/proc/jec_qa_5000_with_reasoning.jsonl'
    output_file = 'datasets/proc/jec_qa_5000_with_reasoning_filtered.jsonl'

    filter_consistent_records(input_file, output_file)
