#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from slime.rollout.rm_hub.math_dapo_utils import compute_score


def load_rows(path: str, limit: int | None):
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    elif path.endswith(".parquet"):
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported eval input: {path}")
    return rows[:limit] if limit else rows


def extract_prompt(row: dict):
    prompt = row.get("prompt") or row.get("input") or row.get("problem")
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError(f"Cannot find prompt in row keys={list(row)}")


def extract_label(row: dict):
    for key in ("label", "answer", "target", "gt"):
        if key in row:
            return row[key]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    rows = load_rows(args.input, args.num_samples)
    prompts = [extract_prompt(row) for row in rows]
    labels = [extract_label(row) for row in rows]

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    results = []
    correct = 0
    total = 0
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="eval"):
        batch_prompts = prompts[start : start + args.batch_size]
        outputs = llm.chat(batch_prompts, sampling)
        for offset, output in enumerate(outputs):
            idx = start + offset
            text = output.outputs[0].text
            reward = compute_score(text, str(labels[idx]))
            score = float(reward["score"])
            acc = bool(reward["acc"])
            pred = reward.get("pred")
            correct += int(acc)
            total += 1
            results.append(
                {
                    "index": idx,
                    "label": labels[idx],
                    "score": score,
                    "acc": acc,
                    "pred": pred,
                    "response": text,
                }
            )

    summary = {
        "model": args.model,
        "input": args.input,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    out = {"summary": summary, "results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
