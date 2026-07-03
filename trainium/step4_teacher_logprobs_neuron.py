# SPDX-License-Identifier: Apache-2.0

"""
Step 4 Phase 2 (Trainium): precompute teacher log-probabilities on Neuron.

Replaces the sglang teacher server + HTTP scoring of the original pipeline
with offline vLLM prompt-logprob scoring: for each row we run the teacher
over (prompt_ids + response_ids) with SamplingParams(prompt_logprobs=0) and
keep the logprob of each actual token for the last len(response_ids)
positions — numerically the same quantity sglang's `input_token_logprobs`
returns.

Input : the *intermediate* parquet from Phase 1
        (data_curation/prepare_lightning_opd.py without --compute-teacher-logprobs),
        columns: prompt (str), label (str), metadata {response_tokens, loss_mask, response}.
Output: <stem>-precomputed.parquet with metadata.teacher_log_probs added —
        byte-compatible with what Step 5 expects.

Resumable: scores in chunks, each chunk saved as a part parquet; re-running
skips completed chunks.

NOTE: requires a Neuron vLLM build that supports `prompt_logprobs`. Verify
with the smoke test in SETUP.md before launching the full run.
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="e.g. Qwen/Qwen3-8B")
    p.add_argument("--tokenizer-path", required=True,
                   help="Student SFT checkpoint (tokenizer must match Phase 1).")
    p.add_argument("--intermediate-parquet", required=True,
                   help="Phase 1 output: <stem>-lightning-opd.parquet")
    p.add_argument("--output-parquet", required=True,
                   help="Final output: <stem>-lightning-opd-precomputed.parquet")
    p.add_argument("--tensor-parallel-size", type=int, default=8,
                   help="NeuronCores for the teacher engine.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="Matches the original teacher server context length.")
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.intermediate_parquet)
    rows = df.to_dict(orient="records")
    print(f"[Step 4] Rows to score: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    parts_dir = Path(args.output_parquet).parent / (Path(args.output_parquet).stem + "-parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    num_chunks = math.ceil(len(rows) / args.chunk_size)

    pending = [c for c in range(num_chunks)
               if not (parts_dir / f"part-{c:05d}.parquet").exists()]
    print(f"[Step 4] {num_chunks} chunks total, {len(pending)} pending")

    if pending:
        print(f"[Step 4] Loading teacher on Neuron: {args.teacher_model} "
              f"(tp={args.tensor_parallel_size})")
        llm = LLM(
            model=args.teacher_model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=True,
        )
        # temperature 0 / no new tokens: pure scoring, same as the sglang call
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

        for c in pending:
            lo, hi = c * args.chunk_size, min((c + 1) * args.chunk_size, len(rows))
            chunk = rows[lo:hi]

            prompts, response_lens = [], []
            for row in chunk:
                meta = row["metadata"]
                prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                response_ids = [int(x) for x in meta["response_tokens"]]
                full_ids = prompt_ids + response_ids
                assert len(full_ids) < args.max_model_len, (
                    f"Sequence of {len(full_ids)} tokens exceeds --max-model-len "
                    f"{args.max_model_len}; increase it and re-run."
                )
                prompts.append(TokensPrompt(prompt_token_ids=full_ids))
                response_lens.append(len(response_ids))

            outputs = llm.generate(prompts, sp)

            chunk_lps = []
            for out, rlen in zip(outputs, response_lens):
                # prompt_logprobs[0] is None (no context); each other entry is
                # {token_id: Logprob} for the *actual* token at that position.
                token_ids = out.prompt_token_ids
                plps = out.prompt_logprobs
                lps = [float(plps[i][token_ids[i]].logprob)
                       for i in range(1, len(token_ids))][-rlen:]
                assert len(lps) == rlen, f"Expected {rlen} logprobs, got {len(lps)}"
                chunk_lps.append(lps)

            pd.DataFrame({"row_idx": list(range(lo, hi)),
                          "teacher_log_probs": chunk_lps}) \
              .to_parquet(parts_dir / f"part-{c:05d}.parquet", index=False)
            print(f"[Step 4] Chunk {c + 1}/{num_chunks} done ({hi}/{len(rows)} rows)")

    # ── Assemble final parquet ───────────────────────────────────────────
    print("[Step 4] Assembling final parquet...")
    all_lps: dict[int, list[float]] = {}
    for c in range(num_chunks):
        part = pd.read_parquet(parts_dir / f"part-{c:05d}.parquet")
        for idx, lps in zip(part["row_idx"], part["teacher_log_probs"]):
            all_lps[int(idx)] = [float(x) for x in lps]
    assert len(all_lps) == len(rows), "Missing chunks — re-run to score them."

    for i, row in enumerate(rows):
        row["metadata"]["teacher_log_probs"] = all_lps[i]

    out = Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[Step 4] Saved {out}")

    # Sanity check, mirroring the original script
    df_check = pd.read_parquet(out)
    meta = df_check.iloc[0]["metadata"]
    print("\n[Step 4] Sanity check row 0:")
    print(f"  prompt[:80]:            {df_check.iloc[0]['prompt'][:80]}")
    print(f"  len(response_tokens):   {len(meta['response_tokens'])}")
    print(f"  len(teacher_log_probs): {len(meta['teacher_log_probs'])}")
    print(f"  teacher_log_probs[:5]:  {meta['teacher_log_probs'][:5]}")


if __name__ == "__main__":
    main()
