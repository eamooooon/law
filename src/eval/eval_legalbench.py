from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys
import json
import time
import os
from collections import Counter
# ==========================================
# 1. 核心配置区
# ==========================================
MODEL_NAME = "sft-qwen2.5-3b-lr2e5-all-2"
MODEL_DIR = f"outputs/{MODEL_NAME}"
# MODEL_DIR = "models/Qwen2.5-3B"
DATASET_PATH = "datasets/eval/legalbench"

# 创建带时间戳的结果目录
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"results/{MODEL_NAME}_{timestamp}"
OUTPUT_JSON = f"{RESULT_DIR}/evaluation_results.json"
OUTPUT_MD = f"{RESULT_DIR}/evaluation_report.md"
RESPONSES_DIR = f"{RESULT_DIR}/responses"

# 【调试开关】如果设为 None，则跑全部 160+ 个任务；如果设为 3，则只跑前 3 个任务用于测试
TEST_LIMIT_TASKS = 10 

print("=========================================")
print("⚖️ LegalBench 全量自动化评测流水线启动")
print("=========================================")

# ==========================================
# 2. 引擎初始化 (全局只执行一次)
# ==========================================
print("\n🚀 [1/3] 正在初始化 vLLM 引擎与 Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        enforce_eager=True
    )
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# 放开 token 限制，给 CoT 充足的思考空间
# sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
sampling_params = SamplingParams(
    temperature=0.7,   # 提高温度：让它每次探索不同的思考路径
    top_p=0.9,         # 配合核采样，截断太离谱的生僻词
    max_tokens=1024
)

# ==========================================
# 3. 获取所有子任务列表
# ==========================================
print("\n📚 [2/3] 正在扫描数据集...")
try:
    all_tasks = get_dataset_config_names(DATASET_PATH, trust_remote_code=True)
    if TEST_LIMIT_TASKS:
        all_tasks = all_tasks[:TEST_LIMIT_TASKS]
        print(f"⚠️ 调试模式开启：仅测试前 {TEST_LIMIT_TASKS} 个任务。")
    print(f"✅ 共发现 {len(all_tasks)} 个子任务。")
except Exception as e:
    print(f"❌ 数据集列表获取失败: {e}")
    sys.exit(1)

# ==========================================
# 4. 开始全量循环评测
# ==========================================
print("\n🧠 [3/3] 开始全量推理与评估...")
start_time = time.time()

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESPONSES_DIR, exist_ok=True)
print(f"📁 结果将保存到: {RESULT_DIR}")

# 用于记录所有任务成绩的字典
results_log = []
total_questions_all = 0
total_correct_all = 0

for task_idx, task_name in enumerate(all_tasks, 1):
    print(f"\n▶️ 进度 [{task_idx}/{len(all_tasks)}] | 当前任务: {task_name}")
    
    # 4.1 加载当前子任务数据
    try:
        dataset = load_dataset(DATASET_PATH, task_name, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ {task_name} 加载失败，已跳过。原因: {e}")
        continue
        
    if len(dataset) == 0:
        continue

    # 4.2 根据任务字段类型自适应组装 Prompt
    cols = dataset.column_names
    prompts = []
    for item in dataset:
        # ── 自适应字段检测 ──────────────────────────────────
        if "choice_0" in cols:                         # 多选题: scalr
            choices = [item[f"choice_{i}"] for i in range(5) if f"choice_{i}" in cols]
            options_text = "\n".join(f"  ({chr(97+i)}) {c}" for i, c in enumerate(choices))
            user_content = (
                f"{item['question']}\n\n"
                f"{options_text}\n\n"
                f"Select the best answer by letter (a-e):"
            )
        elif "question" in cols and "contract" in cols:  # QA题: consumer_contracts_qa
            user_content = (
                f"Contract:\n{item['contract'][:3000]}\n\n"
                f"Question: {item['question']}\n\n"
                f"Answer:"
            )
        elif "question" in cols:                        # 纯问答题: rule_qa, privacy_policy_qa
            user_content = f"{item['question']}\n\nAnswer:"
        elif "slice" in cols:                           # hearsay等: 有slice作为hint
            user_content = (
                f"Context: {item.get('slice', '')}\n"
                f"Text: {item['text']}\n\n"
                f"Answer Yes or No:"
            )
        elif "description" in cols and "statute" in cols: # sara_entailment/numeric
            user_content = (
                f"Statute: {item['statute']}\n"
                f"Description: {item['description']}\n"
                f"Question: {item.get('question', '')}\n\n"
                f"Text: {item['text']}\n\n"
                f"Answer:"
            )
        else:                                           # 默认: text 字段
            user_content = f"{item.get('text', '')}\n\nAnswer:"

        messages = [
            {"role": "system", "content": "You are a highly accurate legal assistant. Think step-by-step, then provide your final answer."},
            {"role": "user", "content": user_content}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # 4.3 批量推理
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False) # 关闭 tqdm 防止终端刷屏

    # 4.4 评分与答案提取
    correct = 0
    for i, output in enumerate(outputs):
        full_pred = output.outputs[0].text.strip().lower()
        truth = str(dataset[i]['answer']).strip().lower()
        
        # 剥离 <think> 标签，提取最终定论
        if "</think>" in full_pred:
            final_answer = full_pred.split("</think>")[-1].strip()
        else:
            final_answer = full_pred
            
        # 兼容性匹配：只要最终答案包含了标准答案即算对
        if truth in final_answer:
            correct += 1

    # 4.5 记录当前任务成绩
    task_acc = correct / len(dataset)
    total_questions_all += len(dataset)
    total_correct_all += correct
    
    print(f"✅ {task_name} 完成 | 准确率: {task_acc * 100:.2f}% ({correct}/{len(dataset)})")
    
    results_log.append({
        "Task Name": task_name,
        "Total Samples": len(dataset),
        "Correct": correct,
        "Accuracy (%)": round(task_acc * 100, 2)
    })

# ==========================================
# 5. 生成报告与汇总分数
# ==========================================
end_time = time.time()
cost_time = (end_time - start_time) / 60

# 5.1 计算两个维度的总分
# Micro-Average (微观平均)：所有题目放在一起算总准确率
micro_acc = (total_correct_all / total_questions_all) * 100 if total_questions_all > 0 else 0

# Macro-Average (宏观平均)：每个任务的准确率加起来求平均 (LegalBench 官方常用此指标)
macro_acc = sum(item["Accuracy (%)"] for item in results_log) / len(results_log) if results_log else 0

# 5.2 组织输出内容
summary = {
    "model_dir": MODEL_DIR,
    "dataset_path": DATASET_PATH,
    "task_count": len(results_log),
    "total_questions": total_questions_all,
    "total_correct": total_correct_all,
    "micro_accuracy_percent": round(micro_acc, 4),
    "macro_accuracy_percent": round(macro_acc, 4),
    "elapsed_minutes": round(cost_time, 4),
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
payload = {
    "summary": summary,
    "tasks": results_log
}

# 5.3 自动创建输出目录
output_dirs = {os.path.dirname(OUTPUT_JSON), os.path.dirname(OUTPUT_MD)}
for output_dir in output_dirs:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

# 5.4 写入 JSON 与 Markdown 报告
try:
    with open(OUTPUT_JSON, mode='w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# LegalBench Evaluation Report",
        "",
        f"- Model: {MODEL_DIR}",
        f"- Dataset: {DATASET_PATH}",
        f"- Generated At: {summary['generated_at']}",
        f"- Elapsed Minutes: {summary['elapsed_minutes']:.2f}",
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
        "|---|---:|---:|---:|"
    ]

    for item in results_log:
        md_lines.append(
            f"| {item['Task Name']} | {item['Total Samples']} | {item['Correct']} | {item['Accuracy (%)']:.2f} |"
        )

    with open(OUTPUT_MD, mode='w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")
except Exception as e:
    print(f"❌ 写入结果文件失败: {e}")
    sys.exit(1)

print("\n=========================================")
print("🏆 【全量评测报告总结】")
print("=========================================")
print(f"⏱️ 总耗时       : {cost_time:.2f} 分钟")
print(f"📚 总评估任务数 : {len(results_log)} 个")
print(f"📝 总答题数量   : {total_questions_all} 道")
print(f"🎯 Micro 准确率 : {micro_acc:.2f}% (按总题数计算)")
print(f"⚖️ Macro 准确率 : {macro_acc:.2f}% (按任务平均计算, 推荐指标)")
print(f"📁 JSON 成绩单   : {os.path.abspath(OUTPUT_JSON)}")
print(f"📄 Markdown 报告 : {os.path.abspath(OUTPUT_MD)}")
print("=========================================")