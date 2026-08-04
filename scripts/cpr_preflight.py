#!/usr/bin/env python3
"""
Step 0 preflight for the CPR experiments.

Two jobs, both cheap and both invalidating everything downstream if they fail:

  1. Assert the four digit action tokens resolve as expected against the *loaded*
     tokenizer. The IDs were read from the vocab, but a different revision or a
     leading-space variant would silently change what the policy is allowed to emit.

  2. Log the UNTRAINED action distribution. Digits carry a far stronger pretrained prior
     than C/D or R/P/S, and trl applies an adaptive KL penalty against the adapter-disabled
     base throughout training. Without this baseline, "the model already prefers 1" and
     "the model learned 1" are indistinguishable in the results.

Deliberately depends only on torch + transformers, not trl, so it runs before the
environment pin in run_cpr.sh is satisfied.

    python scripts/cpr_preflight.py [--model_path google/gemma-2-2b-it] [--adapter_path ...]
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpr_observation_managers import CPRObservationManager, CPRObservationManagerConfig

EXPECTED_TOKENS = {"0": 235276, "1": 235274, "2": 235284, "3": 235304}
PROBE_RESOURCES = [20, 14, 8, 4, 1]


def check_tokens(tokenizer) -> list:
    """Every action string must be exactly one token, with the id the configs assume."""
    print("Action token check")
    action_toks = []
    for string, expected in EXPECTED_TOKENS.items():
        ids = tokenizer.encode(string, add_special_tokens=False)
        assert len(ids) == 1, f"{string!r} is not a single token: {ids}"
        assert ids[0] == expected, f"{string!r} resolved to {ids[0]}, configs assume {expected}"
        action_toks.append(ids[0])
        print(f"  [ok ] {string!r} -> {ids[0]}")
    return action_toks


def action_distribution(model, tokenizer, prompt: str, action_toks: list, device) -> list:
    """Next-token probabilities renormalised over the legal action set."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :].float()
    legal = torch.tensor(action_toks, device=logits.device)
    return torch.softmax(logits.index_select(0, legal), dim=-1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="google/gemma-2-2b-it")
    parser.add_argument("--adapter_path", default=None,
                        help="Optional LoRA adapter; omit to probe the raw base model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    action_toks = check_tokens(tokenizer)

    print(f"\nLoading {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval().to(device)

    manager = CPRObservationManager(
        CPRObservationManagerConfig(
            action_toks=action_toks, action_strings=list(EXPECTED_TOKENS), is_shaper=False, R0=20,
        ),
        n_games=1,
    )

    print(f"\nUntrained action distribution (device: {device})")
    print(f"  {'prompt':<34}" + "".join(f"{s:>9}" for s in EXPECTED_TOKENS))

    reset = manager.game_description + manager.instruction_prompt
    probs = action_distribution(model, tokenizer, reset, action_toks, device)
    print(f"  {'reset (R=20, no history)':<34}" + "".join(f"{p:>9.3f}" for p in probs))

    for R in PROBE_RESOURCES:
        prompt = manager.build_observations([R], [1], [1], [1], [1], inner_t=5, outer_t=0)[0]
        probs = action_distribution(model, tokenizer, prompt, action_toks, device)
        print(f"  {f'R={R}, both requested 1':<34}" + "".join(f"{p:>9.3f}" for p in probs))

    print("\nRecord these numbers before training. Any post-training shift must be read "
          "against this baseline, not against a uniform prior.")


if __name__ == "__main__":
    main()
