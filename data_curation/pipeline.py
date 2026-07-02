# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data curation pipeline: generate responses from a dataset using vLLM.

Each worker (identified by --rank) processes a disjoint shard of the input
dataset, generates responses via vLLM offline inference, and writes results
as Arrow IPC files (one per batch) into a rank-specific output directory.
Checkpointing allows resuming from the last completed batch.

Standalone:
    python data_curation/pipeline.py \
        --model Qwen/Qwen3-4B \
        --input data.jsonl \
        --output-dir output/

Multi-GPU (one model per GPU):
    See run_curation.sh for the recommended launch pattern.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """Load a .jsonl or .parquet dataset into a list of dicts."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        records = df.to_dict("records")
        for record in records:
            if "prompt" in record and hasattr(record["prompt"], "tolist"):
                record["prompt"] = record["prompt"].tolist()
        return records
    elif path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported format: {path}. Use .jsonl or .parquet.")


def save_batch_arrow(rows: list[dict], path: str) -> None:
    """Write a list of dicts as an Arrow IPC file."""
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_curation(args: argparse.Namespace) -> None:
    tag = f"[Rank {args.rank}/{args.world_size}]"

    # ── Load & shard dataset ──────────────────────────────────────────────
    print(f"{tag} Loading dataset: {args.input}")
    dataset = load_dataset(args.input)

    if args.num_samples is not None:
        dataset = dataset[: args.num_samples]
        print(f"{tag} Debug mode: limiting to {args.num_samples} samples")

    if args.world_size > 1:
        dataset = dataset[args.rank :: args.world_size]
    print(f"{tag} Assigned {len(dataset)} samples")

    # ── Output directory ──────────────────────────────────────────────────
    if args.world_size > 1:
        output_dir = Path(args.output_dir) / f"rank{args.rank:05d}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"rank{args.rank:05d}.pkl"

    start_idx = 0
    if ckpt_file.exists():
        with open(ckpt_file, "rb") as f:
            start_idx = pickle.load(f)["next_idx"]
        print(f"{tag} Resuming from index {start_idx}")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"{tag} Loading model: {args.model} (tp={args.tensor_parallel_size})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        dtype=args.dtype,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_responses,
    )

    # ── Batch loop ────────────────────────────────────────────────────────
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    total_saved = 0

    print(f"{tag} Processing {len(dataset)} prompts, batch_size={args.batch_size}, "
          f"total_batches={total_batches}")

    for batch_start in range(start_idx, len(dataset), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        batch_idx = batch_start // args.batch_size

        prompts = [item["prompt"] for item in batch]
        print(f"{tag} Batch {batch_idx + 1}/{total_batches} "
              f"({batch_end - batch_start} samples) ...")

        outputs = llm.chat(prompts, sampling_params)

        # Build results
        rows = []
        for item, output in zip(batch, outputs):
            for completion in output.outputs:
                text = completion.text
                # Ensure <think> tag is present
                if "</think>" in text and not text.strip().startswith("<think>"):
                    text = "<think>\n" + text
                messages = item["prompt"] + [{"role": "assistant", "content": text}]
                rows.append({
                    "messages": messages,
                    "tokens": len(completion.token_ids),
                })

        # Save Arrow file
        arrow_path = output_dir / f"data-{batch_idx:05d}-of-{total_batches:05d}.arrow"
        save_batch_arrow(rows, str(arrow_path))
        total_saved += len(rows)

        # Save checkpoint
        with open(ckpt_file, "wb") as f:
            pickle.dump({"next_idx": batch_end}, f)

        print(f"{tag} Saved {arrow_path.name}  (total: {total_saved})")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if ckpt_file.exists():
        ckpt_file.unlink()
    print(f"{tag} Done! {total_saved} samples → {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate responses from a dataset using vLLM offline inference.",
    )
    # Required
    p.add_argument("--model", type=str, required=True,
                   help="HuggingFace model name or path.")
    p.add_argument("--input", type=str, required=True,
                   help="Input dataset (.jsonl or .parquet).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Root output directory. Each rank writes to a subdirectory.")

    # Generation
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="Max new tokens per response (default: 16384).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7).")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="Nucleus sampling top-p (default: 0.9).")
    p.add_argument("--num-responses", type=int, default=1,
                   help="Number of responses per prompt (default: 1).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Prompts per vLLM batch call (default: 32).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="vLLM GPU memory utilization fraction (default: 0.90).")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="vLLM max model length. Lowering this can improve KV capacity.")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="Maximum concurrent sequences scheduled by vLLM.")
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="Maximum batched tokens scheduled by vLLM.")
    p.add_argument("--dtype", type=str, default="auto",
                   help="vLLM dtype (default: auto). Try bfloat16 on Ampere/Ada/Blackwell.")

    # Parallelism
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="vLLM tensor-parallel size (default: 1).")
    p.add_argument("--rank", type=int, default=None,
                   help="Worker rank (auto-detected from env if omitted).")
    p.add_argument("--world-size", type=int, default=None,
                   help="Total workers (auto-detected from env if omitted).")

    # Misc
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit total samples before sharding (for debugging).")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory for per-rank checkpoint files (default: checkpoints).")

    args = p.parse_args()

    # Auto-detect rank / world_size from environment (torchrun, etc.)
    if args.rank is None:
        args.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if args.world_size is None:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    return args


if __name__ == "__main__":
    run_curation(parse_args())
