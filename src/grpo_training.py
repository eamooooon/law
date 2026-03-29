# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train legal reasoning models with GRPO.
"""
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import request
from urllib.parse import urlparse

import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process must be enclosed within <think> </think> tags. The final answer should appear after the think block, "
    "and using <answer> </answer> tags is optional, i.e., <think> reasoning process here </think>answer here"
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict legal answer judge. Compare the candidate answer against the reference answer. "
    "Return a JSON object with keys 'score' and 'reason'. "
    "The score must be a float between 0.0 and 1.0, where 1.0 means the candidate is fully correct and faithful "
    "to the reference, and 0.0 means it is clearly wrong, contradictory, or missing the core conclusion."
)

JUDGE_USER_PROMPT_TEMPLATE = """Prompt:
{prompt}

Reference answer:
{reference}

Candidate answer:
{candidate}

Scoring rules:
1. Focus on legal correctness, factual consistency, and whether the main conclusion matches the reference.
2. Ignore stylistic differences and minor wording differences.
3. Penalize omissions of the key holding, wrong bottom-line answers, or hallucinated legal conclusions.
4. Return JSON only.
"""


@dataclass
class ScriptArguments:
    """
    Script-specific arguments for GRPO training.
    """

    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face dataset name. Local data/grpo is preferred."},
    )
    train_file_dir: Optional[str] = field(
        default="data/grpo",
        metadata={"help": "Directory containing local GRPO json/jsonl files."},
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Subset name when loading a dataset from the Hugging Face hub."},
    )
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name for hub datasets."})
    preprocessing_num_workers: Optional[int] = field(
        default=10, metadata={"help": "Number of workers for preprocessing"}
    )
    validation_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio when the local directory only provides a train split."},
    )
    prompt_input_field: str = field(default="input", metadata={"help": "Field used as the GRPO user prompt."})
    prompt_instruction_field: str = field(
        default="instruction", metadata={"help": "Optional field used when input is empty."}
    )
    prompt_question_field: str = field(
        default="question", metadata={"help": "Fallback prompt field for older datasets."}
    )
    reference_output_field: str = field(
        default="output", metadata={"help": "Reference output field for judge scoring."}
    )
    reference_answer_field: str = field(
        default="answer", metadata={"help": "Fallback reference field for older datasets."}
    )
    use_only_input_prompt: bool = field(
        default=True,
        metadata={"help": "If true, GRPO prompt uses only the input field and ignores instruction unless input is empty."},
    )
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})

    format_reward_weight: float = field(
        default=1.0, metadata={"help": "Reward for outputs that contain a valid <think> block."}
    )
    think_length_cap: int = field(
        default=2048, metadata={"help": "Visible-character cap for think-length reward."}
    )
    think_length_reward_max: float = field(
        default=1.0, metadata={"help": "Maximum reward contributed by the think-length term."}
    )
    answer_soft_limit: int = field(
        default=256, metadata={"help": "Soft visible-character threshold for answer-length penalty."}
    )
    answer_hard_limit: int = field(
        default=512, metadata={"help": "Hard visible-character threshold for answer-length penalty."}
    )
    answer_over_limit_penalty_max: float = field(
        default=1.0, metadata={"help": "Maximum penalty once answer length reaches the hard threshold."}
    )

    judge_enabled: bool = field(
        default=False, metadata={"help": "Enable LLM judge scoring via a chat-completions compatible API."}
    )
    judge_api_url: Optional[str] = field(
        default=None, metadata={"help": "Judge API base URL or /v1/chat/completions URL."}
    )
    judge_api_key: Optional[str] = field(default=None, metadata={"help": "Judge API key."})
    judge_model: Optional[str] = field(default=None, metadata={"help": "Judge model name."})
    judge_timeout: int = field(default=60, metadata={"help": "Judge request timeout in seconds."})
    judge_temperature: float = field(default=0.0, metadata={"help": "Judge sampling temperature."})
    judge_max_tokens: int = field(default=256, metadata={"help": "Judge max output tokens."})
    judge_score_threshold: float = field(
        default=0.6,
        metadata={"help": "Judge threshold. Samples below this lose their positive think-length reward."},
    )
    judge_reward_weight: float = field(
        default=1.0, metadata={"help": "Weight applied to the judge score before adding it to total reward."}
    )
    judge_default_score_on_error: float = field(
        default=0.0, metadata={"help": "Judge score fallback used when the API call fails."}
    )


def normalize_chat_url(api_url: str) -> str:
    """Allow base URL input and normalize to chat-completions endpoint."""
    raw = (api_url or "").strip()
    if not raw:
        raise ValueError("judge_api_url is empty")

    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw

    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")

    if path.endswith("/v1/chat/completions"):
        return raw

    if path == "":
        return raw.rstrip("/") + "/v1/chat/completions"

    return raw


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Response is not valid JSON object")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def pick_first_nonempty(record: Dict[str, Any], fields: Sequence[str]) -> str:
    for field_name in fields:
        value = stringify(record.get(field_name))
        if value:
            return value
    return ""


def visible_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def extract_think_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if without_think:
        return without_think
    return text.strip()


def has_required_format(text: str) -> bool:
    has_think = bool(re.search(r"<think>.*?</think>", text or "", re.DOTALL | re.IGNORECASE))
    return has_think


def build_prompt_text(record: Dict[str, Any], script_args: ScriptArguments) -> str:
    input_text = stringify(record.get(script_args.prompt_input_field))
    instruction_text = stringify(record.get(script_args.prompt_instruction_field))
    question_text = stringify(record.get(script_args.prompt_question_field))

    if script_args.use_only_input_prompt:
        return input_text or question_text or instruction_text

    parts = [part for part in [instruction_text, input_text or question_text] if part]
    return "\n\n".join(parts).strip()


def collect_json_files(data_dir: str) -> List[str]:
    pattern_json = os.path.join(data_dir, "**", "*.json")
    pattern_jsonl = os.path.join(data_dir, "**", "*.jsonl")
    files = sorted(set(glob.glob(pattern_json, recursive=True) + glob.glob(pattern_jsonl, recursive=True)))
    return [path for path in files if os.path.isfile(path)]


def load_local_dataset(data_dir: str) -> DatasetDict:
    files = collect_json_files(data_dir)
    if not files:
        raise ValueError(f"No json/jsonl files found under local GRPO data dir: {data_dir}")

    split_to_files: Dict[str, List[str]] = {}
    for path in files:
        name = os.path.basename(path).lower()
        if "validation" in name or "valid" in name or "eval" in name:
            split_to_files.setdefault("validation", []).append(path)
        elif "test" in name:
            split_to_files.setdefault("test", []).append(path)
        else:
            split_to_files.setdefault("train", []).append(path)

    if "train" not in split_to_files:
        split_to_files["train"] = list(files)

    logger.info(f"Loading local GRPO files: {split_to_files}")
    return load_dataset("json", data_files=split_to_files)


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4: bool = False, int8: bool = False):
    """Find all linear layer names in the model. reference from qlora paper."""
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


def load_training_datasets(
    script_args: ScriptArguments,
    training_args: GRPOConfig,
) -> Tuple[Dataset, Optional[Dataset]]:
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        raw_datasets = load_local_dataset(script_args.train_file_dir)
    elif script_args.dataset_name:
        logger.warning(f"Falling back to Hugging Face dataset: {script_args.dataset_name}")
        train_dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)
        raw_datasets = DatasetDict({"train": train_dataset})
    else:
        raise ValueError(
            "No GRPO data found. Set --train_file_dir to a directory with json/jsonl files, or pass --dataset_name."
        )

    train_dataset = raw_datasets["train"]
    if script_args.train_samples > 0:
        sample_count = min(script_args.train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42).select(range(sample_count))

    eval_enabled = training_args.eval_strategy != "no"
    eval_dataset = None

    if eval_enabled:
        if "validation" in raw_datasets:
            eval_dataset = raw_datasets["validation"]
        elif "test" in raw_datasets:
            eval_dataset = raw_datasets["test"]
        elif 0.0 < script_args.validation_split_ratio < 1.0 and len(train_dataset) > 1:
            split = train_dataset.train_test_split(test_size=script_args.validation_split_ratio, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            logger.warning("Evaluation is enabled but no validation split is available; eval_dataset will be None.")

    return train_dataset, eval_dataset


class LegalRewardManager:
    def __init__(self, script_args: ScriptArguments):
        self.script_args = script_args
        self.__name__ = "legal_reward_manager"
        self.judge_enabled = bool(script_args.judge_enabled)
        self.judge_api_url = script_args.judge_api_url or os.getenv("GRPO_JUDGE_API_URL", "")
        self.judge_api_key = script_args.judge_api_key or os.getenv("GRPO_JUDGE_API_KEY", "")
        self.judge_model = script_args.judge_model or os.getenv("GRPO_JUDGE_MODEL", "")
        self._judge_cache: Dict[str, float] = {}

        if self.judge_enabled:
            missing = []
            if not self.judge_api_url:
                missing.append("judge_api_url / GRPO_JUDGE_API_URL")
            if not self.judge_model:
                missing.append("judge_model / GRPO_JUDGE_MODEL")
            if missing:
                raise ValueError(f"Judge is enabled, but required settings are missing: {', '.join(missing)}")

    def __call__(self, completions, prompt_text=None, reference_output=None, **kwargs):
        prompt_text = prompt_text or [""] * len(completions)
        reference_output = reference_output or [""] * len(completions)
        rewards: List[float] = []

        for completion, prompt, reference in zip(completions, prompt_text, reference_output):
            content = completion[0]["content"]
            reward, breakdown = self.score_sample(content=content, prompt_text=prompt, reference_output=reference)
            rewards.append(reward)
            logger.debug(f"reward breakdown: {breakdown}")

        return rewards

    def score_sample(self, content: str, prompt_text: str, reference_output: str) -> Tuple[float, Dict[str, Any]]:
        format_ok = has_required_format(content)
        format_reward = self.script_args.format_reward_weight if format_ok else 0.0

        think_text = extract_think_text(content)
        think_len = visible_length(think_text)
        think_length_reward = 0.0
        if think_len > 0 and self.script_args.think_length_cap > 0:
            think_length_reward = (
                min(think_len, self.script_args.think_length_cap) / self.script_args.think_length_cap
            ) * self.script_args.think_length_reward_max

        answer_text = extract_answer_text(content)
        answer_len = visible_length(answer_text)
        answer_length_reward = self.compute_answer_length_reward(answer_len)

        judge_score = None
        judge_reward = 0.0
        judge_passed = True
        if self.judge_enabled:
            judge_score = self.score_with_judge(prompt_text=prompt_text, reference_output=reference_output, answer_text=answer_text)
            judge_reward = judge_score * self.script_args.judge_reward_weight
            judge_passed = judge_score >= self.script_args.judge_score_threshold
            if not judge_passed:
                think_length_reward = 0.0

        total_reward = format_reward + think_length_reward + answer_length_reward + judge_reward
        breakdown = {
            "format_ok": format_ok,
            "format_reward": format_reward,
            "think_len": think_len,
            "think_length_reward": think_length_reward,
            "answer_len": answer_len,
            "answer_length_reward": answer_length_reward,
            "judge_enabled": self.judge_enabled,
            "judge_score": judge_score,
            "judge_passed": judge_passed,
            "judge_reward": judge_reward,
            "total_reward": total_reward,
        }
        return total_reward, breakdown

    def compute_answer_length_reward(self, answer_len: int) -> float:
        soft_limit = self.script_args.answer_soft_limit
        hard_limit = self.script_args.answer_hard_limit
        max_penalty = self.script_args.answer_over_limit_penalty_max

        if soft_limit < 0 or hard_limit < 0:
            return 0.0
        if hard_limit <= soft_limit:
            hard_limit = soft_limit + 1

        if answer_len <= soft_limit:
            return 0.0
        if answer_len >= hard_limit:
            return -max_penalty

        ratio = (answer_len - soft_limit) / (hard_limit - soft_limit)
        return -max_penalty * ratio

    def score_with_judge(self, prompt_text: str, reference_output: str, answer_text: str) -> float:
        cache_key = hashlib.sha1(
            json.dumps(
                {
                    "prompt_text": prompt_text,
                    "reference_output": reference_output,
                    "answer_text": answer_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if cache_key in self._judge_cache:
            return self._judge_cache[cache_key]

        if not reference_output.strip():
            logger.warning("Judge is enabled but reference_output is empty; using fallback score.")
            return self.script_args.judge_default_score_on_error

        try:
            score = self.call_judge_model(prompt_text=prompt_text, reference_output=reference_output, answer_text=answer_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Judge scoring failed: {exc}")
            score = self.script_args.judge_default_score_on_error

        score = max(0.0, min(1.0, float(score)))
        self._judge_cache[cache_key] = score
        return score

    def call_judge_model(self, prompt_text: str, reference_output: str, answer_text: str) -> float:
        chat_url = normalize_chat_url(self.judge_api_url)
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            prompt=prompt_text.strip(),
            reference=reference_output.strip(),
            candidate=answer_text.strip(),
        )
        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.script_args.judge_temperature,
            "max_tokens": self.script_args.judge_max_tokens,
            "response_format": {"type": "json_object"},
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.judge_api_key:
            headers["Authorization"] = f"Bearer {self.judge_api_key}"

        req = request.Request(chat_url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.script_args.judge_timeout) as resp:
            raw = resp.read().decode("utf-8")

        parsed = json.loads(raw)
        choices = parsed.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in judge response: {raw[:500]}")

        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        parsed_content = extract_json_from_text(content)
        score = parsed_content.get("score")
        if score is None:
            raise ValueError(f"Judge response missing 'score': {content}")
        return float(score)


def grpo_train(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    is_main_process = training_args.local_rank in [-1, 0]

    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = load_training_datasets(script_args=script_args, training_args=training_args)

    with training_args.main_process_first(desc="Dataset preparation"):
        def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
            prompt_text = build_prompt_text(record, script_args)
            reference_output = pick_first_nonempty(
                record,
                [script_args.reference_output_field, script_args.reference_answer_field],
            )
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "prompt_text": prompt_text,
                "reference_output": reference_output,
            }

        train_dataset = train_dataset.map(
            preprocess_record,
            num_proc=script_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Processing training dataset" if is_main_process else None,
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_record,
                num_proc=script_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                desc="Processing eval dataset" if is_main_process else None,
            )

    if script_args.judge_enabled:
        non_empty_reference_count = sum(1 for item in train_dataset["reference_output"] if stringify(item))
        if non_empty_reference_count == 0:
            raise ValueError("Judge scoring is enabled, but the dataset does not contain any non-empty reference output.")

    if is_main_process:
        logger.info(f"Prepared train dataset size: {len(train_dataset)}")
        if eval_dataset is not None:
            logger.info(f"Prepared eval dataset size: {len(eval_dataset)}")
        logger.info("*** Initializing model kwargs ***")

    torch_dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    bnb_compute_dtype = torch_dtype if isinstance(torch_dtype, torch.dtype) else torch.bfloat16
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 and QLoRA are currently incompatible.")

    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be enabled")

    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}"
            )
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}"
            )
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    if ddp:
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            usable_mem = int(gpu_props.total_memory * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    if is_main_process and hasattr(model, "hf_device_map"):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, "device"):
                logger.info(f"  {name}: {param.device}")
                break

    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == "all" or (target_modules and "all" in target_modules):
            target_modules = find_all_linear_names(
                model,
                int4=model_args.load_in_4bit,
                int8=model_args.load_in_8bit,
            )
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    reward_manager = LegalRewardManager(script_args)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_manager],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info("*** GRPO Trainer initialized ***")
    logger.debug(f"Trainer: {trainer}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None and is_main_process:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        kwargs = {
            "dataset_name": script_args.dataset_name or script_args.train_file_dir,
            "tags": ["r1", "grpo", "legal"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
