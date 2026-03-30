# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
纯 SFT 监督微调脚本（单轮对话）。

使用 tokenizer.apply_chat_template() 自动适配模型的 chat 格式，
替代旧版 template.py 手写模板，修复 Qwen3 等模型的格式兼容性问题。

支持的数据格式（JSONL）：
    {"instruction": "...", "input": "...", "reasoning": "...", "output": "..."}
    默认情况下 instruction、input、reasoning、output 均为必填。
    当 --include_reasoning False 时，reasoning 可为空或缺失，训练目标仅使用 output。
    注：output_model 字段当前不参与训练。
"""

import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Literal, Optional, Tuple

import torch
import torch.utils.data
from datasets import load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.versions import require_version
from transformers.integrations import is_deepspeed_zero3_enabled

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

# ── ChatML 兜底模板（当 tokenizer 没有内置 chat_template 时使用）──
DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


# ─────────────────────────── 参数定义 ───────────────────────────


@dataclass
class ModelArguments:
    """模型、配置和分词器相关参数。"""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "用于初始化权重的模型检查点。"},
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "是否以 8bit 量化方式加载模型。"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "是否以 4bit 量化方式加载模型。"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "分词器路径；不设置则与模型同路径。"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "预训练模型缓存目录。"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "模型版本（分支名、标签或 commit ID）。"},
    )
    hf_hub_token: Optional[str] = field(
        default=None, metadata={"help": "Hugging Face Hub 访问令牌。"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "是否使用快速分词器。"},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "模型加载精度。Qwen3 等新模型建议 bfloat16。",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "设备映射。"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码。"},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None, metadata={"help": "RoPE 位置编码缩放策略。"}
    )
    flash_attn: Optional[bool] = field(
        default=False, metadata={"help": "是否启用 FlashAttention-2。"}
    )
    shift_attn: Optional[bool] = field(
        default=False, metadata={"help": "是否启用 shifted sparse attention。"}
    )
    neft_alpha: Optional[float] = field(
        default=0, metadata={"help": "NEFTune 噪声 alpha 参数。"}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError(
                "You must specify a valid model_name_or_path to run training."
            )


@dataclass
class DataArguments:
    """训练集和验证集相关参数。"""

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "datasets 库数据集名称。"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "datasets 配置名称。"}
    )
    train_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "训练集路径：单个 json/jsonl 文件或目录。"},
    )
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "验证集路径：单个 json/jsonl 文件或目录。"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "截断训练样本数量（调试用）。"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "截断验证样本数量（调试用）。"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "loss 中是否忽略 pad token。"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖数据集缓存。"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={"help": "无独立验证集时，从训练集切分的验证集百分比。"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "数据预处理进程数。"},
    )
    dataset_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "按比例随机抽样，取值 (0, 1]。"},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("正式训练建议将 max_train_samples 设为 -1 以使用全部样本。")
        if self.dataset_sample_ratio is not None and not (
            0 < self.dataset_sample_ratio <= 1.0
        ):
            raise ValueError("dataset_sample_ratio 必须在 (0, 1] 范围内。")


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "是否启用 PEFT (LoRA)。"})
    train_on_inputs: bool = field(
        default=False, metadata={"help": "是否对输入部分也计算损失。"}
    )
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(
        default=None, metadata={"help": "已有 PEFT 模型路径。"}
    )
    qlora: bool = field(default=False, metadata={"help": "是否启用 QLoRA。"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "最大上下文长度。"},
    )
    system_prompt: Optional[str] = field(
        default="You are a helpful assistant.",
        metadata={"help": "系统提示词，传入 apply_chat_template。"},
    )
    include_reasoning: bool = field(
        default=True,
        metadata={"help": "是否将数据集中的 reasoning 作为 <think>...</think> 训练目标的一部分。"},
    )

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError(
                "You must specify a valid model_max_length >= 60 to run training"
            )


# ─────────────────────────── Trainer & 辅助函数 ───────────────────────────


class SavePeftModelTrainer(Trainer):
    """LoRA 模型保存。"""

    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb

        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if "lm_head" in name:
                continue
            if "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def _collect_data_files(file_dir: str, split_name: Optional[str] = None):
    """收集目录、单文件或逗号分隔输入的 json/jsonl 路径列表。

    规则：
    1. 如果传入目录且包含标准切分文件 train.jsonl / validation.jsonl，
       则按 split_name 优先只取对应文件。
    2. 自动忽略 metadata.json / *_metadata.json 这类辅助文件。
    3. 兼容旧用法：目录递归收集 json/jsonl。
    """
    file_paths = []
    candidates = [item.strip() for item in str(file_dir).split(",") if item.strip()]
    for candidate in candidates:
        if os.path.isfile(candidate):
            base = os.path.basename(candidate).lower()
            if base == "metadata.json" or base.endswith("_metadata.json"):
                continue
            file_paths.append(candidate)
            continue

        if not os.path.isdir(candidate):
            continue

        preferred_files = []
        if split_name == "train":
            preferred_files = [
                os.path.join(candidate, "train.jsonl"),
                os.path.join(candidate, "train.json"),
            ]
        elif split_name == "validation":
            preferred_files = [
                os.path.join(candidate, "validation.jsonl"),
                os.path.join(candidate, "validation.json"),
                os.path.join(candidate, "eval.jsonl"),
                os.path.join(candidate, "eval.json"),
            ]

        matched_preferred = [path for path in preferred_files if os.path.isfile(path)]
        if matched_preferred:
            file_paths.extend(matched_preferred)
            continue

        json_files = glob(f"{candidate}/**/*.json", recursive=True)
        jsonl_files = glob(f"{candidate}/**/*.jsonl", recursive=True)
        for path in json_files + jsonl_files:
            base = os.path.basename(path).lower()
            if base == "metadata.json" or base.endswith("_metadata.json"):
                continue
            if split_name == "train" and base.startswith(("validation.", "eval.", "test.")):
                continue
            if split_name == "validation" and base.startswith(("train.",)):
                continue
            file_paths.append(path)
    return sorted(set(file_paths))


# ─────────────────────────── 主流程 ───────────────────────────


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, script_args = (
            parser.parse_args_into_dataclasses(look_for_args_file=False)
        )

    if training_args.deepspeed is not None:
        training_args.distributed_state.deepspeed_plugin = None

    is_main_process = training_args.local_rank in [-1, 0]

    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
            f"n_gpu: {training_args.n_gpu}, "
            f"distributed: {bool(training_args.local_rank != -1)}, "
            f"16-bits: {training_args.fp16}"
        )

    set_seed(training_args.seed)

    # Prefer native HF NEFTune integration to avoid manual forward hook side effects.
    if model_args.neft_alpha > 0:
        if hasattr(training_args, "neftune_noise_alpha"):
            training_args.neftune_noise_alpha = model_args.neft_alpha
            logger.info(
                f"启用原生 NEFTune: training_args.neftune_noise_alpha={training_args.neftune_noise_alpha}"
            )
        else:
            logger.warning(
                "当前 transformers 版本不支持 neftune_noise_alpha，已忽略 model_args.neft_alpha。"
            )

    # ── 加载分词器 ──
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = (
        model_args.tokenizer_name_or_path or model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, **tokenizer_kwargs
    )

    # Explicitly enforce right padding during training for stable collation/position behavior.
    tokenizer.padding_side = "right"
    tokenizer.init_kwargs["padding_side"] = "right"
    logger.info(f"Set tokenizer.padding_side = {tokenizer.padding_side}")

    # 确保 chat_template 可用
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
        logger.warning(
            "Tokenizer 没有内置 chat_template，使用默认 ChatML 模板。"
            "如果使用非 ChatML 模型，请检查生成的格式是否正确。"
        )

    # 确保特殊 token 存在
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Set pad_token: {tokenizer.pad_token}, "
            f"pad_token_id: {tokenizer.pad_token_id}"
        )

    logger.debug(f"Tokenizer: {tokenizer}")

    IGNORE_INDEX = (
        LabelSmoother.ignore_index
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id
    )

    # ── 加载数据集 ──
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled = raw_datasets["train"].shuffle(seed=42)
            split = shuffled.train_test_split(
                test_size=data_args.validation_split_percentage / 100, seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        data_files = {}
        if data_args.train_file_dir is not None:
            train_data_files = _collect_data_files(
                data_args.train_file_dir, split_name="train"
            )
            if train_data_files:
                logger.info(f"Train files: {train_data_files}")
                data_files["train"] = train_data_files

        if data_args.validation_file_dir is not None:
            val_data_files = _collect_data_files(
                data_args.validation_file_dir, split_name="validation"
            )
            if val_data_files:
                logger.info(f"Validation files: {val_data_files}")
                data_files["validation"] = val_data_files
        elif data_args.train_file_dir is not None:
            auto_val_files = _collect_data_files(
                data_args.train_file_dir, split_name="validation"
            )
            if auto_val_files:
                logger.info(
                    "未显式提供 validation_file_dir，已自动使用同目录验证集文件: "
                    f"{auto_val_files}"
                )
                data_files["validation"] = auto_val_files

        if "train" not in data_files:
            raise ValueError("本地训练需要提供 --train_file_dir")

        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

        if "validation" not in raw_datasets:
            val_pct = (
                data_args.validation_split_percentage / 100.0
                if data_args.validation_split_percentage
                else 0.05
            )
            shuffled_train = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train.train_test_split(test_size=val_pct, seed=42)
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
            logger.warning("未提供独立验证集，已从训练集切分 validation。")

    logger.info(f"Raw datasets: {raw_datasets}")

    # ── 数据预处理 ──
    max_length = script_args.model_max_length
    system_prompt = script_args.system_prompt or ""

    def normalize_token_ids(tokenized):
        """Normalize apply_chat_template return values to list[int]."""
        if isinstance(tokenized, dict):
            ids = tokenized.get("input_ids")
        elif hasattr(tokenized, "input_ids"):
            ids = tokenized.input_ids
        else:
            ids = tokenized

        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        if isinstance(ids, tuple):
            ids = list(ids)

        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]

        if not isinstance(ids, list):
            raise TypeError(f"Unexpected tokenized type: {type(tokenized)}")
        return ids

    def preprocess_function(examples):
        """使用 apply_chat_template 构造训练样本。"""
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        instructions = examples.get("instruction")
        reasonings = examples.get("reasoning")
        outputs = examples.get("output")
        inputs = examples.get("input")
        system_prompts = examples.get("system_prompt")

        if instructions is None or inputs is None or outputs is None:
            raise ValueError(
                "数据集必须包含且提供以下列：'instruction'、'input'、'output'。"
                f" 当前列：{list(examples.keys())}"
            )
        if script_args.include_reasoning and reasonings is None:
            raise ValueError(
                "当前 include_reasoning=True，数据集必须包含 'reasoning' 列。"
                f" 当前列：{list(examples.keys())}"
            )

        reasonings = (
            reasonings if reasonings is not None else [""] * len(instructions)
        )

        if not (len(instructions) == len(inputs) == len(reasonings) == len(outputs)):
            raise ValueError(
                "instruction/input/reasoning/output 四列长度不一致，无法对齐预处理。"
                f"当前列：{list(examples.keys())}"
            )

        def _wrap_with_think_tag(text: str) -> str:
            content = text.strip()
            if content.startswith("<think>") and content.endswith("</think>"):
                content = content[len("<think>") : -len("</think>")].strip()
            return f"<think>\n{content}\n</think>"

        def _get_system_prompt(index):
            if system_prompts is not None:
                if isinstance(system_prompts, list) and index < len(system_prompts):
                    sp = system_prompts[index]
                    if sp:
                        return str(sp).strip()
                elif isinstance(system_prompts, str) and system_prompts:
                    return system_prompts.strip()
            return system_prompt

        for i, (instruction, input_text, reasoning_text, output_text) in enumerate(
            zip(instructions, inputs, reasonings, outputs)
        ):
            instruction = str(instruction).strip() if instruction else ""
            input_text = str(input_text).strip() if input_text else ""
            reasoning_text = str(reasoning_text).strip() if reasoning_text else ""
            output_text = str(output_text).strip() if output_text else ""

            if not instruction or not input_text or not output_text:
                raise ValueError(
                    "发现缺失必填字段的样本：instruction/input/output 均不能为空。"
                )
            if script_args.include_reasoning and not reasoning_text:
                raise ValueError(
                    "当前 include_reasoning=True，发现 reasoning 为空的样本。"
                )

            # include_reasoning=True: 训练目标为 <think>reasoning</think> + output
            # include_reasoning=False: 训练目标仅使用 output
            assistant_text = (
                f"{_wrap_with_think_tag(reasoning_text)}\n{output_text}"
                if script_args.include_reasoning
                else output_text
            )

            user_query = (
                f"{instruction}\n{input_text}" if input_text else instruction
            )
            sys_prompt = _get_system_prompt(i)

            # 构建完整对话
            messages_full = []
            if sys_prompt:
                messages_full.append({"role": "system", "content": sys_prompt})
            messages_full.append({"role": "user", "content": user_query})
            messages_full.append({"role": "assistant", "content": assistant_text})

            # 构建 prompt-only（用于确定 label 遮罩位置）
            messages_prompt = []
            if sys_prompt:
                messages_prompt.append({"role": "system", "content": sys_prompt})
            messages_prompt.append({"role": "user", "content": user_query})

            full_ids = tokenizer.apply_chat_template(
                messages_full, tokenize=True, add_generation_prompt=False
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages_prompt, tokenize=True, add_generation_prompt=True
            )
            full_ids = normalize_token_ids(full_ids)
            prompt_ids = normalize_token_ids(prompt_ids)

            # 截断：保留完整 prompt，从 response 尾部截断
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            prompt_len = min(len(prompt_ids), len(full_ids))

            # 极端超长输入场景：截断后若整段都落在 prompt 范围内，
            # 则该样本不会产生有效监督信号，直接丢弃。
            if prompt_len >= len(full_ids):
                continue

            # 构建 labels
            if script_args.train_on_inputs:
                labels = list(full_ids)
            else:
                labels = [IGNORE_INDEX] * prompt_len + list(full_ids[prompt_len:])

            input_ids_list.append(full_ids)
            attention_mask_list.append([1] * len(full_ids))
            labels_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=labels_list,
        )

    def filter_empty_labels(example):
        return not all(label == IGNORE_INDEX for label in example["labels"])

    def sample_dataset_by_ratio(dataset, split_name):
        ratio = data_args.dataset_sample_ratio
        if ratio is None or ratio >= 1.0:
            return dataset
        sample_size = max(1, int(len(dataset) * ratio)) if len(dataset) > 0 else 0
        logger.info(
            f"抽样 {split_name}: ratio={ratio}, seed={training_args.seed}, "
            f"原始={len(dataset)}, 抽样后={sample_size}"
        )
        if sample_size == 0:
            return dataset
        return dataset.shuffle(seed=training_args.seed).select(range(sample_size))

    # ── 处理训练集 ──
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = sample_dataset_by_ratio(
            raw_datasets["train"], "train"
        ).shuffle(seed=training_args.seed)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if len(train_dataset) == 0:
            raise ValueError(
                "训练集在原始加载/抽样后为空。请检查 --train_file_dir、dataset_sample_ratio、max_train_samples。"
            )

        if is_main_process and len(train_dataset) > 0:
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="训练集分词"):
            tokenized_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="训练集分词处理中" if is_main_process else None,
            )
            train_dataset = tokenized_dataset.filter(
                filter_empty_labels, num_proc=data_args.preprocessing_num_workers
            )

            if len(train_dataset) == 0:
                raise ValueError(
                    "训练集在预处理后为空：所有样本可能被过滤。"
                    "请检查数据是否包含且非空 instruction/input/reasoning/output，"
                    "并确认截断后不全是 prompt。"
                )

            if is_main_process:
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("训练集分词示例:")
                logger.debug(
                    f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}"
                )
                replaced_labels = [
                    label if label != IGNORE_INDEX else tokenizer.pad_token_id
                    for label in list(train_dataset[0]["labels"])
                ]
                logger.debug(
                    f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}"
                )

    # ── 处理验证集 ──
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="验证集分词"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = sample_dataset_by_ratio(
                raw_datasets["validation"], "validation"
            )
            max_eval_samples = len(eval_dataset)
            if (
                data_args.max_eval_samples is not None
                and data_args.max_eval_samples > 0
            ):
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            if eval_size > 500:
                logger.warning(
                    f"验证集较大: {eval_size}，训练会变慢，"
                    f"考虑使用 --max_eval_samples=50"
                )
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="验证集分词处理中",
            )
            eval_dataset = eval_dataset.filter(
                filter_empty_labels, num_proc=data_args.preprocessing_num_workers
            )
            if len(eval_dataset) == 0:
                raise ValueError(
                    "验证集在预处理后为空：请检查 validation 数据字段和内容。"
                )
            logger.debug(f"Num eval_samples (after filter): {len(eval_dataset)}")

    # ── 加载模型 ──
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = None
        if model_args.device_map in ["None", "none", ""]:
            model_args.device_map = None
        if script_args.qlora and (
            len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()
        ):
            logger.warning("FSDP / DeepSpeed ZeRO-3 与 QLoRA 不兼容。")

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )

        # RoPE 缩放
        if model_args.rope_scaling is not None:
            if hasattr(config, "rope_scaling"):
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if (
                    current_max_length
                    and script_args.model_max_length > current_max_length
                ):
                    scaling_factor = float(
                        math.ceil(script_args.model_max_length / current_max_length)
                    )
                else:
                    logger.warning(
                        f"model_max_length({script_args.model_max_length}) <= "
                        f"max_position_embeddings({current_max_length})"
                    )
                    scaling_factor = 1.0
                setattr(
                    config,
                    "rope_scaling",
                    {"type": model_args.rope_scaling, "factor": scaling_factor},
                )
                logger.info(
                    f"RoPE scaling: {model_args.rope_scaling}, factor={scaling_factor}"
                )
            else:
                logger.warning("当前模型不支持 RoPE scaling。")

        # FlashAttention-2
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2.")
            else:
                logger.warning("FlashAttention-2 未安装。")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("建议使用 --flash_attn 配合 shift_attn。")

        # Shifted sparse attention
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("Using shifted sparse attention (group_size_ratio=1/4).")
            else:
                logger.warning("当前模型不支持 shifted sparse attention。")

        # 量化配置
        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        quantization_config = None
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit 和 load_in_8bit 不能同时启用。")
        elif load_in_8bit or load_in_4bit:
            logger.info(
                f"量化加载: load_in_4bit={load_in_4bit}, load_in_8bit={load_in_8bit}"
            )
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 与量化不兼容。")
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": model_args.trust_remote_code,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "device_map": model_args.device_map,
        }

        # 多 GPU auto device_map
        num_gpus = torch.cuda.device_count()
        if model_args.device_map == "auto":
            if num_gpus > 1 and not ddp:
                model_kwargs["device_map"] = "auto"
                max_memory = {}
                for i in range(num_gpus):
                    gpu_props = torch.cuda.get_device_properties(i)
                    usable_mem = int(gpu_props.total_memory * 0.8)
                    max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
                model_kwargs["max_memory"] = max_memory

        logger.info(f"Model kwargs: {model_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        logger.info("模型加载完成")

        # 打印模型分布
        if is_main_process:
            if hasattr(model, "hf_device_map") and model.hf_device_map:
                device_count = {}
                for device in model.hf_device_map.values():
                    ds = str(device)
                    device_count[ds] = device_count.get(ds, 0) + 1
                logger.info(f"设备映射统计: {device_count}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    total = (
                        torch.cuda.get_device_properties(i).total_memory / 1024**3
                    )
                    logger.info(
                        f"GPU {i}: 已分配={allocated:.1f}GB, 总计={total:.1f}GB"
                    )

        # ChatGLM / InternLM2 兼容
        if getattr(config, "model_type", None) in ("chatglm", "internlm2"):
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # MoE 兼容
        if (
            getattr(config, "model_type", None) == "mixtral"
            and is_deepspeed_zero3_enabled()
        ):
            require_version("deepspeed>=0.13.0", "pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules
            from transformers.models.mixtral.modeling_mixtral import (
                MixtralSparseMoeBlock,
            )

            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        if (
            getattr(config, "model_type", None) == "deepseek_v3"
            and is_deepspeed_zero3_enabled()
        ):
            require_version("deepspeed>=0.13.0", "pip install deepspeed>=0.13.0")
            for layer in model.model.layers:
                if "DeepseekV3MoE" in str(type(layer.mlp)):
                    layer.mlp._z3_leaf = True
    else:
        raise ValueError("必须指定 model_name_or_path")

    # ── LoRA / 全参数 ──
    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA (PEFT)")

        output_layer = getattr(model, "lm_head")
        if (
            isinstance(output_layer, torch.nn.Linear)
            and output_layer.weight.dtype != torch.float32
        ):

            def fp32_forward_post_hook(
                module: torch.nn.Module,
                args: Tuple[torch.Tensor],
                output: torch.Tensor,
            ):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        if script_args.peft_path is not None:
            logger.info(f"加载已有 PEFT: {script_args.peft_path}")
            model = PeftModel.from_pretrained(
                model, script_args.peft_path, is_trainable=True
            )
        else:
            logger.info("初始化新 PEFT 模型")
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(
                    model, training_args.gradient_checkpointing
                )
            target_modules = (
                script_args.target_modules.split(",")
                if script_args.target_modules
                else None
            )
            if target_modules and "all" in target_modules:
                target_modules = find_all_linear_names(
                    model, int4=load_in_4bit, int8=load_in_8bit
                )
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(",")
            logger.info(f"LoRA target_modules: {target_modules}")
            logger.info(f"LoRA rank: {script_args.lora_rank}")
            logger.info(f"modules_to_save: {modules_to_save}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters")
        model = model.float()
        print_trainable_parameters(model)

    # ── 配置 Trainer ──
    if training_args.gradient_checkpointing and getattr(
        model, "supports_gradient_checkpointing", False
    ):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    )

    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── 训练 ──
    if training_args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(
                f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}"
            )
            replaced_labels = [
                label if label != IGNORE_INDEX else tokenizer.pad_token_id
                for label in sample["labels"][0]
            ]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            ckpt = training_args.resume_from_checkpoint
            if isinstance(ckpt, str):
                if ckpt.lower() == "true":
                    checkpoint = True
                elif ckpt.lower() == "false":
                    checkpoint = None
                else:
                    checkpoint = ckpt
            else:
                checkpoint = ckpt
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True
        tokenizer.padding_side = "left"
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # ── 评估 ──
    if training_args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
