# SPDX-License-Identifier: Apache-2.0

"""
Step 4 Phase 2 (Trainium) — FORWARD-PASS teacher scoring (the working path).

Why this exists instead of the vLLM scorer
-------------------------------------------
`step4_teacher_logprobs_neuron.py` scores teacher logprobs with vLLM
`prompt_logprobs`. That does NOT work on the vllm-neuron 0.16 plugin:
  * on-device sampling (the default) never returns logprobs — you get [None];
  * the CPU-sampling path (`NEURON_ON_DEVICE_SAMPLING_DISABLED=1`) is broken in
    this release — a `Sampler` import bug (`from vllm.v1.sample import sampler
    as Sampler` then `Sampler()`), and even after patching that, the model
    runner still routes to `_sample_on_device`, which crashes with
    `hidden_states[reorder_indices] -> IndexError`.
So we skip vLLM entirely and compute teacher logprobs the same way step 5
computes STUDENT logprobs: one forward pass through NeuronModelForCausalLM, then
`token_logprobs()` over the returned per-position logits. No sampling involved,
so none of the plugin's sampling bugs apply.

This runs in the TRAINING venv (optimum-neuron), launched with torchrun across
NeuronCores exactly like step 5 — NOT in the vLLM inference venv.

I/O is byte-compatible with the vLLM scorer: reads the Phase-1 intermediate
parquet, writes <stem>-precomputed.parquet with metadata.teacher_log_probs
added. Resumable via per-chunk part parquets.

Parallelism: tensor-parallel across all NeuronCores, data-parallel = 1 (one
Trainium2 chip). Every rank runs the same forward and — because token_logprobs
all-reduces over the TP group — ends up with identical logprobs; only rank 0
writes. On multi-chip boxes set --tensor-parallel-size < NUM_CORES to get DP,
but this script currently shards work by rank only when dp_size>1 (see below).

VALIDATE ON HARDWARE before a full run: this mirrors step 5's model handling,
but forward-only-without-a-Trainer and static-shape padding are the parts most
likely to need a tweak on your SDK. Run it on a --num-rows 8 slice first.
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer

from optimum.neuron import NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

from logprob_utils import token_logprobs

RANK = int(os.environ.get("RANK", "0"))
IS_MAIN = RANK == 0
DEBUG = os.environ.get("OPD_DEBUG", "0") == "1"


def _log(*a):
    if IS_MAIN:
        print("[Step 4/forward]", *a, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="e.g. Qwen/Qwen3-8B")
    p.add_argument("--tokenizer-path", required=True,
                   help="Student SFT checkpoint (tokenizer must match Phase 1).")
    p.add_argument("--intermediate-parquet", required=True,
                   help="Phase 1 output: <stem>-lightning-opd.parquet")
    p.add_argument("--output-parquet", required=True,
                   help="Final output: <stem>-lightning-opd-precomputed.parquet")
    p.add_argument("--tensor-parallel-size", type=int, default=4,
                   help="NeuronCores for the teacher (4 = one trn2.3xlarge chip).")
    p.add_argument("--max-seq-len", type=int, default=8192,
                   help="Static padded length. Must be >= the longest "
                        "prompt+response; the script checks and tells you.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Sequences per forward (static). Keep small — the teacher "
                        "LM head emits [B, max_seq_len, vocab] logits.")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Rows per resumable part parquet.")
    p.add_argument("--num-rows", type=int, default=None,
                   help="Debug: only score the first N rows.")
    return p.parse_args()


def build_full_ids(row, tokenizer):
    """(full_ids, response_len) — identical construction to the vLLM scorer."""
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    response_ids = [int(x) for x in row["metadata"]["response_tokens"]]
    return prompt_ids + response_ids, len(response_ids)


def score_microbatch(model, seqs, pad_id, max_seq_len, vocab_size, device):
    """Score a list of (full_ids, rlen) → list of per-response-token logprobs.

    Pads to a fixed (batch_size, max_seq_len) so Neuron compiles one shape. The
    batch is right-padded; response tokens sit just before the pad, so their
    logprobs come from the last `rlen` valid next-token positions.
    """
    B = len(seqs)
    input_ids = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_seq_len), dtype=torch.long)
    lengths = []
    for b, (full_ids, _rlen) in enumerate(seqs):
        n = len(full_ids)
        input_ids[b, :n] = torch.tensor(full_ids, dtype=torch.long)
        attention_mask[b, :n] = 1
        lengths.append(n)

    # The model lives on the Neuron (XLA) device; inputs must too, or the
    # tensor-parallel collectives raise "Expected XLA tensor. Got: CPU...".
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    # Position i predicts token i+1 (same alignment as step 5's compute_loss).
    lp = token_logprobs(logits[:, :-1, :], input_ids[:, 1:], vocab_size)  # [B, L-1]
    xm.mark_step()          # force the lazy XLA graph to execute
    lp = lp.float().cpu()

    out = []
    for b, (full_ids, rlen) in enumerate(seqs):
        n = lengths[b]
        # Response tokens occupy positions [n-rlen, n); the logprob of the token
        # at position p is lp[p-1]. So the response logprobs are lp[n-rlen-1 : n-1].
        row_lp = lp[b, n - rlen - 1: n - 1].tolist()
        assert len(row_lp) == rlen, f"got {len(row_lp)} logprobs, expected {rlen}"
        out.append([float(x) for x in row_lp])
    return out


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    df = pd.read_parquet(args.intermediate_parquet)
    rows = df.to_dict(orient="records")
    if args.num_rows is not None:
        rows = rows[: args.num_rows]
    _log(f"Rows to score: {len(rows)}")

    # The Qwen3 flash-attention NKI kernel requires the padded sequence length to
    # be a multiple of 2048 ("Only support sequence as multiples of 2K"), so round
    # max_seq_len up to the next 2K boundary.
    BLK = 2048
    if args.max_seq_len % BLK != 0:
        rounded = ((args.max_seq_len + BLK - 1) // BLK) * BLK
        _log(f"Rounding --max-seq-len {args.max_seq_len} up to {rounded} "
             f"(flash-attn kernel needs a multiple of {BLK}).")
        args.max_seq_len = rounded

    # Pre-tokenize and enforce the static length up front (before the slow load).
    full = [build_full_ids(r, tokenizer) for r in rows]
    max_full = max((len(f) for f, _ in full), default=0)
    if max_full > args.max_seq_len:
        need = ((max_full + BLK - 1) // BLK) * BLK
        raise SystemExit(
            f"[Step 4/forward] Longest prompt+response is {max_full} tokens but "
            f"--max-seq-len is {args.max_seq_len}. Re-run with "
            f"--max-seq-len {need} (it resumes from finished chunks)."
        )
    _log(f"Longest sequence {max_full} tokens fits max-seq-len {args.max_seq_len}.")

    parts_dir = Path(args.output_parquet).parent / (Path(args.output_parquet).stem + "-parts")
    if IS_MAIN:
        parts_dir.mkdir(parents=True, exist_ok=True)
    num_chunks = math.ceil(len(rows) / args.chunk_size)
    pending = [c for c in range(num_chunks)
               if not (parts_dir / f"part-{c:05d}.parquet").exists()]
    _log(f"{num_chunks} chunks total, {len(pending)} pending")

    if pending:
        # trn_config carries the tensor-parallel layout; build it via
        # NeuronTrainingArguments exactly like step 5, then load the model with
        # it (this shards the weights and sets up the TP process groups that
        # token_logprobs() reduces over).
        training_args = NeuronTrainingArguments(
            output_dir=str(parts_dir / "_trn_tmp"),
            do_train=False,
            bf16=True,
            per_device_train_batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            report_to=[],
        )
        _log(f"Loading teacher: {args.teacher_model} (tp={args.tensor_parallel_size})")
        device = xm.xla_device()
        model = NeuronModelForCausalLM.from_pretrained(
            args.teacher_model,
            training_args.trn_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = model.to(device)  # move weights to Neuron (no Trainer to do it)
        model.eval()
        vocab_size = model.config.vocab_size

        for c in pending:
            lo, hi = c * args.chunk_size, min((c + 1) * args.chunk_size, len(rows))
            chunk = full[lo:hi]

            chunk_lps = []
            for s in range(0, len(chunk), args.batch_size):
                mb = chunk[s: s + args.batch_size]
                # Pad the final micro-batch up to batch_size with a repeat of its
                # last row so Neuron keeps a single compiled (B, L) shape; drop
                # the padding rows afterwards.
                pad_n = args.batch_size - len(mb)
                mb_padded = mb + [mb[-1]] * pad_n if pad_n else mb
                lps = score_microbatch(model, mb_padded, pad_id, args.max_seq_len,
                                       vocab_size, device)
                chunk_lps.extend(lps[: len(mb)])

            if DEBUG and IS_MAIN and c == pending[0]:
                first = chunk_lps[0]
                flat = [x for r in chunk_lps for x in r]
                bad = sum(1 for x in flat if x != x or x == float("-inf"))
                _log(f"[DEBUG] row {lo}: kept {len(first)} logprobs; "
                     f"chunk mean={sum(flat)/len(flat):.4f}, min={min(flat):.4f}, "
                     f"max={max(flat):.4f}, nan/-inf={bad} (should be <= 0)")

            if IS_MAIN:
                pd.DataFrame({"row_idx": list(range(lo, hi)),
                              "teacher_log_probs": chunk_lps}) \
                  .to_parquet(parts_dir / f"part-{c:05d}.parquet", index=False)
            _log(f"Chunk {c + 1}/{num_chunks} done ({hi}/{len(rows)} rows)")

    # ── Assemble final parquet (rank 0 only) ─────────────────────────────────
    if not IS_MAIN:
        return
    _log("Assembling final parquet...")
    all_lps = {}
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
    _log(f"Saved {out}")

    meta = pd.read_parquet(out).iloc[0]["metadata"]
    _log("Sanity check row 0:")
    _log(f"  len(response_tokens):   {len(meta['response_tokens'])}")
    _log(f"  len(teacher_log_probs): {len(meta['teacher_log_probs'])}")
    _log(f"  teacher_log_probs[:5]:  {meta['teacher_log_probs'][:5]}")


if __name__ == "__main__":
    main()
