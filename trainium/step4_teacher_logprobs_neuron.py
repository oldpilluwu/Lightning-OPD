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

V1 CAVEAT (vLLM-Neuron plugin >= 0.5, SDK >= 2.29): the plugin defaults to
*on-device sampling*, which only surfaces temperature/top_k/top_p and does NOT
return logprobs — so `prompt_logprobs` comes back empty/unsupported. To score,
on-device sampling must be disabled so vLLM samples on CPU from the full logits.
Do that by passing an override via --override-neuron-config (JSON) or the
OVERRIDE_NEURON_CONFIG env var, e.g.
    OVERRIDE_NEURON_CONFIG='{"on_device_sampling_config": null}'
Confirm the exact key against your plugin version with the SETUP.md §6(b) smoke
test before the full run; if CPU sampling still won't return prompt_logprobs,
use the forward-pass fallback (SETUP.md §10).
"""

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

DEBUG = os.environ.get("OPD_DEBUG", "0") == "1"


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
                   help="NeuronCores for the teacher engine (4 = one trn2.3xlarge chip).")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="Matches the original teacher server context length.")
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--override-neuron-config", type=str,
                   default=os.environ.get("OVERRIDE_NEURON_CONFIG", ""),
                   help="JSON dict forwarded to LLM(override_neuron_config=...). "
                        "Needed on the V1 Neuron plugin to disable on-device "
                        "sampling so prompt_logprobs are returned (CPU sampling). "
                        "See the module docstring.")
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
        # Fail fast BEFORE the expensive model load: vLLM's generate needs at
        # least one output token (max_tokens=1), so a scored sequence must fit
        # the context with one token of headroom, i.e. len(full_ids) <=
        # max_model_len - 1. (The original sglang path used max_new_tokens=0 and
        # could score a context-filling sequence; on vLLM we need the headroom.)
        # Check the whole dataset up front so we don't abort mid-run after
        # scoring several chunks.
        max_full = 0
        for row in rows:
            meta = row["metadata"]
            n = (len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                 + len(meta["response_tokens"]))
            max_full = max(max_full, n)
        if max_full >= args.max_model_len:
            raise SystemExit(
                f"[Step 4] Longest prompt+response is {max_full} tokens, but "
                f"--max-model-len is {args.max_model_len} and vLLM needs "
                f"len(full_ids) <= max_model_len - 1. Re-run with "
                f"--max-model-len >= {max_full + 1} (it resumes from finished chunks)."
            )
        print(f"[Step 4] Longest sequence {max_full} tokens fits max-model-len "
              f"{args.max_model_len}.")

        llm_kwargs = dict(
            model=args.teacher_model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            # See pipeline_neuron.py: V1 prefix caching is on by default and then
            # requires an explicit block_size. Scoring gains nothing from it.
            enable_prefix_caching=os.environ.get("ENABLE_PREFIX_CACHING", "0") == "1",
            trust_remote_code=True,
        )
        if args.override_neuron_config.strip():
            try:
                onc = json.loads(args.override_neuron_config)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[Step 4] --override-neuron-config is not valid JSON: {e}")
            # The vllm-neuron plugin reads this from additional_config, NOT as a
            # direct LLM() kwarg (neuronx_distributed_model_loader.py:1050).
            llm_kwargs["additional_config"] = {"override_neuron_config": onc}
            print(f"[Step 4] additional_config = {llm_kwargs['additional_config']}")
        print(f"[Step 4] Loading teacher on Neuron: {args.teacher_model} "
              f"(tp={args.tensor_parallel_size})")
        llm = LLM(**llm_kwargs)
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
                # Defensive: the up-front scan already guaranteed this fits.
                assert len(full_ids) < args.max_model_len, (
                    f"Sequence of {len(full_ids)} tokens does not leave room for "
                    f"vLLM's 1 output token within --max-model-len {args.max_model_len}; "
                    f"re-run with --max-model-len >= {len(full_ids) + 1}."
                )
                prompts.append(TokensPrompt(prompt_token_ids=full_ids))
                response_lens.append(len(response_ids))

            outputs = llm.generate(prompts, sp)

            # Fail loudly if the build ignored prompt_logprobs. On the V1 Neuron
            # plugin this is the on-device-sampling default: generation succeeds
            # but out.prompt_logprobs is None. Detect it on the first output
            # instead of crashing on a NoneType subscript deeper in the loop.
            if outputs and (outputs[0].prompt_logprobs is None
                            or all(p is None for p in outputs[0].prompt_logprobs)):
                raise SystemExit(
                    "[Step 4] The teacher returned no prompt_logprobs — this "
                    "Neuron vLLM build isn't surfacing them (on-device sampling "
                    "is likely enabled). Disable it so vLLM samples on CPU, e.g.\n"
                    "    OVERRIDE_NEURON_CONFIG='{\"on_device_sampling_config\": null}' \\\n"
                    "        bash trainium/step4_precompute_teacher_logprobs.sh ...\n"
                    "Verify the exact key with the SETUP.md §6(b) smoke test. If "
                    "CPU sampling still won't return them, use the forward-pass "
                    "fallback (SETUP.md §10). No chunks were written, so nothing "
                    "to clean up before retrying."
                )

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

            if DEBUG and c == pending[0] and outputs:
                # Alignment proof on the first scored row: the logprob we keep
                # for response position j must be the logprob of the *response*
                # token itself, and the slice must cover exactly the last rlen
                # positions of the full sequence.
                out0, rlen0 = outputs[0], response_lens[0]
                token_ids = out0.prompt_token_ids
                total = len(token_ids)
                resp_ids = token_ids[total - rlen0:]
                lp0 = chunk_lps[0]
                print(f"[Step 4][DEBUG] row {lo}: prompt+resp tokens={total}, "
                      f"response tokens={rlen0}, kept logprobs={len(lp0)}")
                print(f"[Step 4][DEBUG]   first response token id={resp_ids[0]} "
                      f"({tokenizer.decode([resp_ids[0]])!r}), teacher_lp={lp0[0]:.4f}")
                print(f"[Step 4][DEBUG]   last response token id={resp_ids[-1]} "
                      f"({tokenizer.decode([resp_ids[-1]])!r}), teacher_lp={lp0[-1]:.4f}")
                import statistics
                flat = [x for row_lp in chunk_lps for x in row_lp]
                bad = sum(1 for x in flat if x != x or x == float("-inf"))  # NaN/-inf
                print(f"[Step 4][DEBUG]   chunk logprob stats: "
                      f"mean={statistics.fmean(flat):.4f}, min={min(flat):.4f}, "
                      f"max={max(flat):.4f}, nan/-inf count={bad} "
                      f"(logprobs should be <= 0)")

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
