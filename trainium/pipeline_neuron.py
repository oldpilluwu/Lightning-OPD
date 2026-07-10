# SPDX-License-Identifier: Apache-2.0

"""
Trainium port of data_curation/pipeline.py: generate responses with vLLM
running on AWS Neuron (NxD Inference backend) instead of CUDA.

Identical semantics to the original:
  * each worker (--rank) processes a disjoint shard (dataset[rank::world_size])
  * responses sampled with temperature 0.7 / top-p 0.9 / max 16384 new tokens
  * output rows {"messages": [...], "tokens": int} written as Arrow IPC files
  * per-rank pickle checkpoint allows resuming from the last completed batch

Differences (Neuron-specific, does not change the data):
  * `tensor_parallel_size` counts NeuronCores (not GPUs)
  * `max_model_len` / `max_num_seqs` are fixed at engine build time because
    Neuron compiles static graphs; override via --max-model-len/--max-num-seqs
  * worker device isolation is done by the launcher via NEURON_RT_VISIBLE_CORES
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from vllm import LLM, SamplingParams

DEBUG = os.environ.get("OPD_DEBUG", "0") == "1"


def _dbg(tag, *args):
    if DEBUG:
        print(f"{tag} [DEBUG]", *args, flush=True)


def _describe_prompt(prompt):
    """One-line shape of a prompt: chat-message list vs raw string."""
    if isinstance(prompt, list):
        roles = [m.get("role", "?") for m in prompt if isinstance(m, dict)]
        return f"list[{len(prompt)}] roles={roles}"
    return f"{type(prompt).__name__} len={len(prompt) if hasattr(prompt, '__len__') else '?'}"


def load_dataset(path: str) -> list[dict]:
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
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def run_curation(args: argparse.Namespace) -> None:
    tag = f"[Rank {args.rank}/{args.world_size}]"

    print(f"{tag} Loading dataset: {args.input}")
    dataset = load_dataset(args.input)

    if args.num_samples is not None:
        dataset = dataset[: args.num_samples]
        print(f"{tag} Debug mode: limiting to {args.num_samples} samples")

    if args.world_size > 1:
        dataset = dataset[args.rank :: args.world_size]
    print(f"{tag} Assigned {len(dataset)} samples")

    if DEBUG and dataset and args.rank == 0:
        first = dataset[0]
        _dbg(tag, f"input record keys: {sorted(first.keys())}")
        _dbg(tag, f"prompt structure: {_describe_prompt(first.get('prompt'))}")
        _dbg(tag, f"prompt[0] repr: {str(first.get('prompt'))[:400]}")

    if args.world_size > 1:
        output_dir = Path(args.output_dir) / f"rank{args.rank:05d}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"rank{args.rank:05d}.pkl"

    start_idx = 0
    if ckpt_file.exists():
        with open(ckpt_file, "rb") as f:
            start_idx = pickle.load(f)["next_idx"]
        print(f"{tag} Resuming from index {start_idx}")

    # Qwen3 advertises a ~131K native context. Letting NxD Inference inherit
    # that value makes compilation and KV-cache allocation needlessly large for
    # this workload. The paper only needs prompt + 16K generated tokens, so pin
    # both vLLM and the underlying Neuron config to the requested static shape.
    override_neuron_config = {
        "max_context_length": args.max_model_len,
        "seq_len": args.max_model_len,
        "ctx_batch_size": 1,
        "batch_size": args.max_num_seqs,
        # One exact graph is substantially faster to compile than a bucket set.
        "enable_bucketing": False,
    }
    print(f"{tag} Loading model on Neuron: {args.model} "
          f"(tp={args.tensor_parallel_size}, effective context cap="
          f"{args.max_model_len}, concurrency={args.max_num_seqs}, bucketing=off)")
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        # Prefix caching is disabled below, so the vllm-neuron plugin uses a
        # contiguous KV cache and requires exactly one block per sequence.
        num_gpu_blocks_override=args.max_num_seqs,
        additional_config={"override_neuron_config": override_neuron_config},
        # vLLM V1 enables prefix caching by default, which then asserts that an
        # explicit block_size is set ("When prefix caching is enabled, block_size
        # must be set"). Batch generation over ~unique prompts gains nothing from
        # it, so disable it. Set ENABLE_PREFIX_CACHING=1 to opt back in (you must
        # then also pass a block_size via extra engine args).
        enable_prefix_caching=os.environ.get("ENABLE_PREFIX_CACHING", "0") == "1",
        trust_remote_code=True,
    )
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    llm = LLM(**llm_kwargs)

    engine = getattr(llm, "llm_engine", None)
    model_config = getattr(engine, "model_config", None)
    effective_max_len = getattr(model_config, "max_model_len", None)
    if effective_max_len is not None and effective_max_len != args.max_model_len:
        raise RuntimeError(
            f"vLLM ignored the requested context cap: expected "
            f"{args.max_model_len}, got {effective_max_len}. Refusing to compile "
            "the model's full native context."
        )
    print(f"{tag} vLLM effective max_model_len="
          f"{effective_max_len if effective_max_len is not None else args.max_model_len}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_responses,
    )

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

        if DEBUG and batch_idx == 0 and args.rank == 0 and outputs:
            comp = outputs[0].outputs[0]
            tok_counts = [len(o.outputs[0].token_ids) for o in outputs]
            _dbg(tag, f"batch 0: {len(outputs)} prompts, generated token counts "
                      f"min/mean/max = {min(tok_counts)}/"
                      f"{sum(tok_counts) / len(tok_counts):.0f}/{max(tok_counts)}")
            _dbg(tag, f"sample completion tokens={len(comp.token_ids)}, "
                      f"finish_reason={comp.finish_reason}, "
                      f"has_</think>={'</think>' in comp.text}")
            _dbg(tag, f"sample completion text[:500]: {comp.text[:500]!r}")

        rows = []
        for item, output in zip(batch, outputs):
            for completion in output.outputs:
                text = completion.text
                if "</think>" in text and not text.strip().startswith("<think>"):
                    text = "<think>\n" + text
                messages = item["prompt"] + [{"role": "assistant", "content": text}]
                rows.append({
                    "messages": messages,
                    "tokens": len(completion.token_ids),
                })

        arrow_path = output_dir / f"data-{batch_idx:05d}-of-{total_batches:05d}.arrow"
        save_batch_arrow(rows, str(arrow_path))
        total_saved += len(rows)

        with open(ckpt_file, "wb") as f:
            pickle.dump({"next_idx": batch_end}, f)

        print(f"{tag} Saved {arrow_path.name}  (total: {total_saved})")

    if ckpt_file.exists():
        ckpt_file.unlink()
    print(f"{tag} Done! {total_saved} samples -> {output_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate responses with vLLM on AWS Neuron (Trainium).",
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    # Generation — identical defaults to data_curation/pipeline.py
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-responses", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)

    # Neuron engine sizing (compiled statically)
    p.add_argument("--max-model-len", type=int,
                   default=int(os.environ.get("MAX_MODEL_LEN", 18432)),
                   help="Engine context length: prompt + max-tokens must fit.")
    p.add_argument("--max-num-seqs", type=int,
                   default=int(os.environ.get("MAX_NUM_SEQS", 8)),
                   help="Continuous-batching slots compiled into the engine.")
    p.add_argument("--num-gpu-blocks-override", type=int, default=None,
                   help="vLLM scheduler KV-block count. With Neuron contiguous KV "
                        "cache and prefix caching disabled, set this exactly to "
                        "--max-num-seqs.")

    # Parallelism
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="Number of NeuronCores per worker.")
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--world-size", type=int, default=None)

    # Misc
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    args = p.parse_args()

    if args.rank is None:
        args.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if args.world_size is None:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    return args


if __name__ == "__main__":
    run_curation(parse_args())
