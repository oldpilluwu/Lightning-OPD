# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare Lightning OPD parquet from student rollout data.

Phase 1 – tokenize (CPU-friendly):
    Reads student rollout parquet, builds prompt via chat template,
    tokenizes responses, truncates to --max-response-len, writes intermediate
    parquet WITHOUT teacher logprobs.

Phase 2 – precompute teacher logprobs (requires GPU / teacher sglang server):
    Reads the intermediate parquet produced in Phase 1, sends each
    (prompt + response) sequence to the teacher sglang server, stores
    per-token response logprobs back into the metadata, writes the final
    parquet.

Usage (Phase 1, CPU node):
    python3 data_curation/prepare_lightning_opd.py \\
        --tokenizer-path checkpoints/sft \\
        --input-parquet data/rollouts/rollouts.parquet \\
        --output-dir data/lightning_opd

Usage (Phase 2, GPU node with teacher sglang running):
    python3 data_curation/prepare_lightning_opd.py \\
        --tokenizer-path checkpoints/sft \\
        --input-parquet data/rollouts/rollouts.parquet \\
        --output-dir data/lightning_opd \\
        --compute-teacher-logprobs \\
        --teacher-url http://127.0.0.1:13141/generate
"""

import argparse
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Lightning OPD parquet data (tokenize + optional teacher logprobs)."
    )
    parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="Path to HuggingFace tokenizer (e.g. the student SFT checkpoint).",
    )
    parser.add_argument(
        "--input-parquet", type=str, required=True,
        help="Path to student rollout parquet. Expected columns: messages (list[dict]), tokens (int).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where intermediate and final parquet files are written.",
    )
    parser.add_argument(
        "--max-response-len", type=int, default=4096,
        help="Maximum response token length; longer responses are truncated (default: 4096).",
    )
    parser.add_argument(
        "--compute-teacher-logprobs", action="store_true",
        help="Run Phase 2: compute teacher logprobs via a running sglang server.",
    )
    parser.add_argument(
        "--teacher-url", type=str, default="http://127.0.0.1:13141/generate",
        help="Teacher sglang server URL (default: http://127.0.0.1:13141/generate).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=64,
        help="Number of concurrent requests to teacher sglang server (default: 64).",
    )
    return parser.parse_args()


# ── Phase 1: tokenize ────────────────────────────────────────────────────────

def phase1_tokenize(args, intermediate_path: Path):
    print(f"[Phase 1] Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"[Phase 1] Loading input parquet: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"[Phase 1] Total rows: {len(df)}")

    rows_out = []
    truncated = 0
    skipped = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="Tokenizing"):
        messages = row.messages

        user_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_str = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        assistant_msg = None
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_msg = msg["content"]
                break
        if assistant_msg is None:
            skipped += 1
            continue

        response_ids = tokenizer.encode(assistant_msg, add_special_tokens=False)

        if len(response_ids) > args.max_response_len:
            truncated += 1
            response_ids = response_ids[:args.max_response_len]
            assistant_msg = tokenizer.decode(response_ids, skip_special_tokens=False)

        rows_out.append({
            "prompt": prompt_str,
            "label": "0",
            "metadata": {
                "is_lightning_opd": True,
                "response_tokens": response_ids,
                "loss_mask": [1] * len(response_ids),
                "response": assistant_msg,
            },
        })

    print(f"[Phase 1] Rows written: {len(rows_out)}, "
          f"truncated to {args.max_response_len}: {truncated}, skipped: {skipped}")
    df_out = pd.DataFrame(rows_out)
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(intermediate_path, index=False)
    print(f"[Phase 1] Saved to {intermediate_path}")


# ── Phase 2: precompute teacher logprobs ─────────────────────────────────────

async def _fetch_logprobs(
    session: aiohttp.ClientSession,
    teacher_url: str,
    full_ids: list[int],
    response_len: int,
) -> list[float]:
    """Call teacher sglang server and return per-token logprobs for the response portion."""
    payload = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with session.post(teacher_url, json=payload) as resp:
        resp.raise_for_status()
        ret = await resp.json()

    all_lps = ret["meta_info"]["input_token_logprobs"]
    response_lps = [float(item[0]) for item in all_lps[1:]][-response_len:]
    assert len(response_lps) == response_len, (
        f"Expected {response_len} logprobs, got {len(response_lps)}"
    )
    return response_lps


async def _process_all(args, tokenizer, rows: list[dict]) -> list[list[float]]:
    """Process all rows concurrently with a live progress bar, preserving order."""
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    results = [None] * len(rows)

    async def bounded_fetch(idx: int, full_ids: list[int], response_len: int):
        async with semaphore:
            result = await _fetch_logprobs(session, args.teacher_url, full_ids, response_len)
        results[idx] = result
        pbar.update(1)

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(rows), desc="[Phase 2] Teacher logprobs") as pbar:
            tasks = []
            for idx, row in enumerate(rows):
                meta = row["metadata"]
                prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                response_ids = [int(x) for x in meta["response_tokens"]]
                full_ids = prompt_ids + response_ids
                tasks.append(bounded_fetch(idx, full_ids, len(response_ids)))
            await asyncio.gather(*tasks)

    return results


def phase2_logprobs(args, intermediate_path: Path, output_path: Path):
    print(f"[Phase 2] Loading intermediate parquet: {intermediate_path}")
    df = pd.read_parquet(intermediate_path)
    rows = df.to_dict(orient="records")
    print(f"[Phase 2] Total rows: {len(rows)}")

    print(f"[Phase 2] Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"[Phase 2] Computing teacher logprobs via {args.teacher_url} "
          f"(concurrency={args.concurrency})")
    all_logprobs = asyncio.run(_process_all(args, tokenizer, rows))

    for row, lps in zip(rows, all_logprobs):
        row["metadata"]["teacher_log_probs"] = lps

    df_out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"[Phase 2] Saved to {output_path}")

    # Sanity check
    df_check = pd.read_parquet(output_path)
    row0 = df_check.iloc[0]
    meta = row0["metadata"]
    print("\n[Phase 2] Sanity check row 0:")
    print(f"  prompt[:80]:              {row0['prompt'][:80]}")
    print(f"  label:                    {row0['label']}")
    print(f"  len(response_tokens):     {len(meta['response_tokens'])}")
    print(f"  len(teacher_log_probs):   {len(meta['teacher_log_probs'])}")
    print(f"  teacher_log_probs[:5]:    {meta['teacher_log_probs'][:5]}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    input_stem = Path(args.input_parquet).stem
    intermediate_path = output_dir / f"{input_stem}-lightning-opd.parquet"
    output_path = output_dir / f"{input_stem}-lightning-opd-precomputed.parquet"

    if args.compute_teacher_logprobs:
        if not intermediate_path.exists():
            print("[INFO] Intermediate parquet not found, running Phase 1 first.")
            phase1_tokenize(args, intermediate_path)
        phase2_logprobs(args, intermediate_path, output_path)
    else:
        phase1_tokenize(args, intermediate_path)
        print(f"\n[INFO] To add teacher logprobs, re-run with --compute-teacher-logprobs "
              f"after starting the teacher sglang server.")


if __name__ == "__main__":
    main()
