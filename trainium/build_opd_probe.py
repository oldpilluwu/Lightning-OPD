# SPDX-License-Identifier: Apache-2.0

"""
Build a fixed OPD probe set for monitoring SFT convergence.

The teacher generates responses on a random sample of the OPD prompts
(DAPO-Math-17k), then scores its own tokens in a second temperature-0 pass
(prompt_logprobs, identical semantics to step 4 teacher scoring — we do NOT
reuse generation-time logprobs because those can reflect the
temperature-scaled sampling distribution).

Output parquet has the same schema as the step-4 precomputed parquet:
    prompt (str), label (str), metadata {response_tokens, loss_mask,
    teacher_log_probs, response}
so the training-time probe (opd_probe.py) — and any future tooling — can
read either file interchangeably.

Runs in the INFERENCE venv (vLLM on Neuron).
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True, help="e.g. Qwen/Qwen3-8B")
    p.add_argument("--opd-prompts", required=True,
                   help="OPD prompt dataset (.jsonl), e.g. dapo-math-17k.jsonl")
    p.add_argument("--output", required=True, help="Output probe parquet path.")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--max-response-len", type=int, default=4096,
                   help="Matches the OPD max response length.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    if out.exists():
        print(f"[Probe] {out} already exists — keeping the existing probe set "
              f"(delete it to rebuild; a frozen probe keeps metrics comparable).")
        return

    with open(args.opd_prompts) as f:
        records = [json.loads(line) for line in f]
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.num_prompts, len(records)))
    print(f"[Probe] Sampled {len(sample)} / {len(records)} OPD prompts (seed {args.seed})")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    # Render prompts ourselves so the stored prompt string is exactly the
    # context used for generation AND for later probe scoring. Qwen3 student
    # and teacher share the same tokenizer/template (the whole pipeline
    # relies on a shared vocab).
    prompt_strs = [
        tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        for item in sample
    ]

    print(f"[Probe] Loading teacher on Neuron: {args.teacher_model} "
          f"(tp={args.tensor_parallel_size})")
    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )

    # ── Pass 1: teacher generates responses (pipeline sampling params) ──
    gen_sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_response_len,
    )
    gen_out = llm.generate(prompt_strs, gen_sp)

    # ── Pass 2: temperature-0 scoring of the generated tokens ───────────
    score_inputs, response_ids_all = [], []
    for prompt_str, o in zip(prompt_strs, gen_out):
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        response_ids = list(o.outputs[0].token_ids)[: args.max_response_len]
        response_ids_all.append(response_ids)
        score_inputs.append(TokensPrompt(prompt_token_ids=prompt_ids + response_ids))

    score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    score_out = llm.generate(score_inputs, score_sp)

    rows = []
    for prompt_str, o, response_ids, gen in zip(
        prompt_strs, score_out, response_ids_all, gen_out
    ):
        token_ids = o.prompt_token_ids
        plps = o.prompt_logprobs
        lps = [float(plps[i][token_ids[i]].logprob)
               for i in range(1, len(token_ids))][-len(response_ids):]
        assert len(lps) == len(response_ids)
        rows.append({
            "prompt": prompt_str,
            "label": "0",
            "metadata": {
                "is_lightning_opd": True,
                "response_tokens": response_ids,
                "loss_mask": [1] * len(response_ids),
                "teacher_log_probs": lps,
                "response": gen.outputs[0].text,
            },
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)

    n_tok = sum(len(r["metadata"]["response_tokens"]) for r in rows)
    t_nll = -sum(sum(r["metadata"]["teacher_log_probs"]) for r in rows) / max(n_tok, 1)
    print(f"[Probe] Saved {len(rows)} rows ({n_tok} response tokens) -> {out}")
    print(f"[Probe] Teacher per-token NLL on its own samples (entropy baseline): {t_nll:.4f}")


if __name__ == "__main__":
    main()
