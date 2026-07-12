# SPDX-License-Identifier: Apache-2.0
"""Validate eager and compiled native generation against an fp32 CPU reference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from trainium.sft_data_generation_native.pipeline import (
    generate_static,
    import_torchneuron,
    load_model_and_tokenizer,
    render_prompt,
)


def run_phase(args: argparse.Namespace, device_name: str, mode: str) -> list[int]:
    phase_args = argparse.Namespace(**vars(args))
    phase_args.device = device_name
    phase_args.mode = mode
    phase_args.dtype = "float32" if device_name == "cpu" else "bfloat16"
    device = torch.device(device_name)
    if device_name == "neuron":
        import_torchneuron()
    model, tokenizer, dtype = load_model_and_tokenizer(phase_args, device)
    rendered = render_prompt(tokenizer, [{"role": "user", "content": args.prompt}])
    encoded = tokenizer(
        [rendered],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.prefill_bucket,
    )
    eos = tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(item) for item in eos}
    result = generate_static(
        model,
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        eos_token_ids=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        dtype=dtype,
    )[0]
    del model
    gc.collect()
    return result


def token_match(reference: list[int], candidate: list[int]) -> float:
    denominator = max(len(reference), len(candidate), 1)
    matches = sum(left == right for left, right in zip(reference, candidate))
    return matches / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt", default="What is 2 + 2? Explain briefly.")
    parser.add_argument("--prefill-bucket", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--compile-backend", default="neuron")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--output", default="agent_artifacts/traces/native_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    reference = run_phase(args, "cpu", "eager")
    eager = run_phase(args, "neuron", "eager")
    eager_match = token_match(reference, eager)
    result = {
        "model": args.model,
        "criterion": "greedy token match >= 0.95",
        "cpu_fp32_tokens": reference,
        "neuron_eager_bf16_tokens": eager,
        "neuron_eager_match": eager_match,
    }
    if eager_match < 0.95:
        raise SystemExit(f"eager validation failed: greedy token match={eager_match:.2%}")
    if not args.skip_compile:
        compiled = run_phase(args, "neuron", "compile")
        compiled_match = token_match(reference, compiled)
        result["neuron_compiled_bf16_tokens"] = compiled
        result["neuron_compiled_match"] = compiled_match
        if compiled_match < 0.95:
            raise SystemExit(f"compiled validation failed: greedy token match={compiled_match:.2%}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"validation passed -> {output}")


if __name__ == "__main__":
    main()
