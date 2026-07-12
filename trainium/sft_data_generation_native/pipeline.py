# SPDX-License-Identifier: Apache-2.0
"""Generate Lightning-OPD SFT data with Transformers on native TorchNeuron.

This module deliberately does not use vLLM, XLA, or NxD.  It follows the
TorchNeuron bring-up order: eager execution on ``torch.device("neuron")`` first,
then an optional ``torch.compile(backend="neuron")`` performance phase.  A CPU
run is available for validation and as an explicitly requested fallback.

The decode loop uses a fixed prompt bucket, a preallocated ``StaticCache``, and
a fixed-size attention mask.  Consequently every decode call has the same
tensor shapes, as required by the Beta-3 native compile backend.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def import_torchneuron() -> str:
    """Import the package that registers the native ``neuron`` device."""
    configured = os.environ.get("TORCHNEURON_IMPORT")
    candidates = [configured] if configured else []
    candidates.extend(["torch_neuronx", "torch_neuron", "torchneuron"])
    errors: list[str] = []
    for candidate in dict.fromkeys(item for item in candidates if item):
        try:
            importlib.import_module(candidate)
            return candidate
        except Exception as exc:  # environment discovery; model errors remain unwrapped
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "No native TorchNeuron backend could be imported. Set TORCHNEURON_IMPORT "
        "to the import name supplied by the Beta-3 image. Attempts: " + "; ".join(errors)
    )


def load_dataset(path: str) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        import pandas as pd

        records = pd.read_parquet(path).to_dict("records")
        for record in records:
            prompt = record.get("prompt")
            if hasattr(prompt, "tolist"):
                record["prompt"] = prompt.tolist()
        return records
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Unsupported format: {path}. Use .jsonl or .parquet.")


def save_batch_arrow(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def render_prompt(tokenizer: Any, prompt: Any) -> str:
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    if isinstance(prompt, str):
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        return prompt
    raise TypeError(f"Expected prompt to be a string or chat-message list, got {type(prompt).__name__}")


def top_p_sample_cpu(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample on CPU in fp32 while leaving the expensive forward on Neuron."""
    logits = torch.nan_to_num(logits.detach().float().cpu(), nan=-1e9, posinf=1e9, neginf=-1e9)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative - sorted_probs <= top_p
    keep[..., 0] = True
    sorted_probs = sorted_probs * keep
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    selected = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, selected).squeeze(-1)


@torch.no_grad()
def generate_static(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: set[int],
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[list[int]]:
    from transformers import StaticCache

    batch_size, prefill_len = input_ids.shape
    total_len = prefill_len + max_new_tokens
    cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=total_len,
        device=device,
        dtype=dtype,
    )
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
    attention_mask[:, :prefill_len] = prompt_mask.to(torch.long)
    position_ids = (prompt_mask.long().cumsum(-1) - 1).clamp_min(0)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask[:, :prefill_len],
        position_ids=position_ids,
        past_key_values=cache,
        cache_position=torch.arange(prefill_len, device=device),
        use_cache=True,
    )
    next_logits = output.logits[:, -1, :]
    prompt_lengths = prompt_mask.long().sum(-1)
    generated = [[] for _ in range(batch_size)]
    finished_cpu = torch.zeros(batch_size, dtype=torch.bool)

    for step in range(max_new_tokens):
        next_cpu = top_p_sample_cpu(next_logits, temperature, top_p)
        next_cpu = torch.where(finished_cpu, torch.full_like(next_cpu, pad_token_id), next_cpu)
        for row, token in enumerate(next_cpu.tolist()):
            if not finished_cpu[row]:
                generated[row].append(token)
        finished_cpu |= torch.tensor([token in eos_token_ids for token in next_cpu.tolist()])
        if bool(finished_cpu.all()):
            break

        position = prefill_len + step
        active = (~finished_cpu).to(device=device, dtype=torch.long)
        attention_mask[:, position] = active
        output = model(
            input_ids=next_cpu.to(device).unsqueeze(-1),
            attention_mask=attention_mask,
            position_ids=(prompt_lengths + step).unsqueeze(-1),
            past_key_values=cache,
            cache_position=torch.tensor([position], device=device),
            use_cache=True,
        )
        next_logits = output.logits[:, -1, :]
    return generated


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, torch.dtype]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    if args.dtype == "float32":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device=device).eval()
    print(f"[model] loaded {args.model} on {device} as {dtype} in {time.monotonic() - started:.1f}s")
    if args.mode == "compile":
        print(f"[compile] backend={args.compile_backend} dynamic=False fullgraph={args.fullgraph}")
        model = torch.compile(
            model,
            backend=args.compile_backend,
            dynamic=False,
            fullgraph=args.fullgraph,
        )
    return model, tokenizer, dtype


def run_curation(args: argparse.Namespace) -> None:
    tag = f"[rank {args.rank}/{args.world_size}]"
    requested_device = torch.device(args.device)
    if requested_device.type == "neuron":
        backend_import = import_torchneuron()
        print(f"{tag} native backend import: {backend_import}")

    dataset = load_dataset(args.input)
    if args.num_samples is not None:
        dataset = dataset[: args.num_samples]
    dataset = dataset[args.rank :: args.world_size]
    print(f"{tag} assigned {len(dataset)} prompts")
    if not dataset:
        return

    output_dir = Path(args.output_dir)
    if args.world_size > 1:
        output_dir /= f"rank{args.rank:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"rank{args.rank:05d}.pkl"
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, "rb") as handle:
            start_idx = int(pickle.load(handle)["next_idx"])
        print(f"{tag} resuming at local sample {start_idx}")

    model, tokenizer, dtype = load_model_and_tokenizer(args, requested_device)
    eos_ids = tokenizer.eos_token_id
    eos_token_ids = {int(eos_ids)} if isinstance(eos_ids, int) else {int(item) for item in eos_ids}
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    for batch_start in range(start_idx, len(dataset), args.batch_size):
        real_batch = dataset[batch_start : batch_start + args.batch_size]
        padded_batch = list(real_batch)
        while len(padded_batch) < args.batch_size:
            padded_batch.append(real_batch[-1])
        rendered = [render_prompt(tokenizer, item["prompt"]) for item in padded_batch]
        unpadded = tokenizer(rendered, add_special_tokens=False, truncation=False)
        prompt_lengths = [len(token_ids) for token_ids in unpadded["input_ids"]]
        if max(prompt_lengths) > args.prefill_bucket:
            raise ValueError(
                f"prompt has {max(prompt_lengths)} tokens but --prefill-bucket is "
                f"{args.prefill_bucket}; increase the bucket instead of truncating paper data"
            )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.prefill_bucket,
        )
        started = time.monotonic()
        generated = generate_static(
            model,
            encoded["input_ids"].to(requested_device),
            encoded["attention_mask"].to(requested_device),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            device=requested_device,
            dtype=dtype,
        )
        rows = []
        for item, token_ids in zip(real_batch, generated):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            if "</think>" in text and not text.strip().startswith("<think>"):
                text = "<think>\n" + text
            prompt_messages = item["prompt"]
            if isinstance(prompt_messages, str):
                prompt_messages = [{"role": "user", "content": prompt_messages}]
            rows.append(
                {
                    "messages": list(prompt_messages) + [{"role": "assistant", "content": text}],
                    "tokens": len(token_ids),
                }
            )

        batch_idx = batch_start // args.batch_size
        arrow_path = output_dir / f"data-{batch_idx:05d}-of-{total_batches:05d}.arrow"
        save_batch_arrow(rows, arrow_path)
        next_idx = min(batch_start + args.batch_size, len(dataset))
        with open(checkpoint_file, "wb") as handle:
            pickle.dump({"next_idx": next_idx}, handle)
        print(
            f"{tag} batch {batch_idx + 1}/{total_batches}: {len(rows)} rows -> "
            f"{arrow_path.name} ({time.monotonic() - started:.1f}s)"
        )

    checkpoint_file.unlink(missing_ok=True)
    print(f"{tag} complete: {len(dataset)} rows -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["neuron", "cpu"], default="neuron")
    parser.add_argument("--mode", choices=["eager", "compile"], default="eager")
    parser.add_argument("--compile-backend", default=os.environ.get("TORCHNEURON_BACKEND", "neuron"))
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--prefill-bucket", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-responses", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--checkpoint-dir", default="agent_artifacts/data/curation_checkpoints")
    args = parser.parse_args()
    if args.num_responses != 1:
        parser.error("native static generation currently supports --num-responses 1")
    if min(args.prefill_bucket, args.max_tokens, args.batch_size, args.world_size) <= 0:
        parser.error("prefill bucket, max tokens, batch size, and world size must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must satisfy 0 <= rank < world-size")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed + args.rank)
    try:
        run_curation(args)
    except Exception:
        if not args.allow_cpu_fallback or args.device == "cpu":
            raise
        print("\n*** NEURON EXECUTION FAILED; RETRYING ON CPU EAGER (FALLBACK, NOT DEVICE SUCCESS) ***\n")
        args.device = "cpu"
        args.mode = "eager"
        args.dtype = "float32"
        run_curation(args)


if __name__ == "__main__":
    main()
