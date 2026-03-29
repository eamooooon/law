from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys

# ==========================================
# 1. 路径配置
# ==========================================
# MODEL_DIR = "models/Qwen2.5-3B"
MODEL_DIR = "outputs/sft-qwen2.5-3b-lr2e5-all"
DATASET_PATH = "datasets/eval/legalbench"
TASK_NAME = "abercrombie"

print("=========================================")
print(f"⚖️ 开始本地评测 LegalBench 子任务: {TASK_NAME}")
print("=========================================")

print("📚 1. 加载数据集...")
try:
    dataset = load_dataset(DATASET_PATH, TASK_NAME, split="test", trust_remote_code=True)
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    sys.exit(1)

print("🚀 2. 初始化 vLLM 与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

llm = LLM(
    model=MODEL_DIR,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    enforce_eager=True
)

# 5 类商标分类
LABELS = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]

print("🛠️ 3. 正在使用 Chat Template 组装 Prompt...")
prompts = []
for item in dataset:
    text = item['text']
    messages = [
        {"role": "system", "content": "You are a highly accurate legal assistant."},
        {"role": "user", "content": (
            f'Classify the trademark description below into EXACTLY ONE of the following categories: '
            f'{", ".join(LABELS)}.\n\n'
            f'Trademark description: "{text}"\n\n'
            f'Category (only one word):'
        )}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(prompt)

sampling_params = SamplingParams(temperature=0.0, max_tokens=1024) 

print(f"🧠 4. 开始批量推理 {len(prompts)} 条数据 (等待时间会稍长，因为它在思考)...")
outputs = llm.generate(prompts, sampling_params)

# 5. 评分
correct = 0
print("\n🔍 【模型输出透视 (前 5 条)】")

for i, output in enumerate(outputs):
    full_pred = output.outputs[0].text.strip().lower()
    truth = str(dataset[i]['answer']).strip().lower()
    
    # 核心提取逻辑：尝试把 <think> 标签里的内容切掉
    if "</think>" in full_pred:
        # 只取 </think> 标签之后的内容作为最终答案
        final_answer = full_pred.split("</think>")[-1].strip()
    else:
        # 如果模型想得太久，连 1024 个 token 都没想完（没生成闭合标签），就兜底用全部文本
        final_answer = full_pred
    
    if i < 5:
        print(f"--- 样本 {i+1} ---")
        print(f"🎯 标准答案 (Truth): {truth}")
        print(f"🤔 思考过程 : {full_pred.split('</think>')[0]}... (已折叠)")
        print(f"🤖 最终提取 (Pred) : {final_answer}")
    
    # 在提取出的最终答案中进行匹配
    if truth in final_answer:
        correct += 1

print("\n📊 【最终成绩】")
print(f"🎯 任务 [{TASK_NAME}] 准确率: {correct / len(dataset) * 100:.2f}% ({correct}/{len(dataset)})")
print("⚠️ 提示: 如果结尾出现 EngineCore died 报错，请直接忽略，推理已完成。")

del llm
import gc
gc.collect()