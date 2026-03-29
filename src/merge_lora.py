import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path.")
    parser.add_argument("--adapter_model", type=str, required=True, help="LoRA adapter path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Merged model output path.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for loading.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = resolve_dtype(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_model)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
