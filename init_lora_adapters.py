"""Initialize rank-2 LoRA adapters for ShapeLLM reproduction (paper Appendix A.3)."""

import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.device_utils import get_device_str


def create_adapter(
    model_path: str,
    output_dir: str,
    rank: int = 2,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> None:
    target_modules = target_modules or ["q_proj", "v_proj"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Saving adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Use adapter_path: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create rank-2 LoRA adapters for google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/gemma-2-2b-it",
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="adapter/gemma-2-2b-it-r2",
        help="Directory to save a single adapter",
    )
    parser.add_argument(
        "--ipd_pair",
        action="store_true",
        help="Create opponent + shaper adapter dirs for IPD shaping repro",
    )
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()

    print(f"Using device: {get_device_str()}")

    if args.ipd_pair:
        create_adapter(args.model_path, "adapter/ipd_opponent_r2", rank=args.rank)
        create_adapter(args.model_path, "adapter/ipd_shaper_r2", rank=args.rank)
    else:
        create_adapter(args.model_path, args.output_dir, rank=args.rank)


if __name__ == "__main__":
    main()