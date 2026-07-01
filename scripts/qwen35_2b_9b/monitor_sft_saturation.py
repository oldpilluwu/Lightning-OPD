#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int:
    match = CKPT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def is_checkpoint_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    if not (path / "trainer_state.json").exists():
        return False
    has_weights = any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").exists()
    return has_weights


def load_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs) if xs else math.nan


def selected_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def eval_checkpoint(args, checkpoint: Path, probe_rows: list[dict]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    student_nlls = []
    teacher_nlls = []
    gaps = []
    token_counts = []

    for row in tqdm(probe_rows, desc=f"probe {checkpoint.name}", leave=False):
        prompt_ids = [int(x) for x in row["prompt_tokens"]]
        response_ids = [int(x) for x in row["response_tokens"]]
        teacher_lps = [float(x) for x in row["teacher_log_probs"]]
        if not response_ids or len(response_ids) != len(teacher_lps):
            continue

        input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
        log_probs = selected_logprobs(model, input_ids).squeeze(0)
        start = len(prompt_ids) - 1
        response_lp = log_probs[start : start + len(response_ids)]

        student_nll = -response_lp.mean().item()
        teacher_nll = -mean(teacher_lps)
        student_nlls.append(student_nll)
        teacher_nlls.append(teacher_nll)
        gaps.append(student_nll - teacher_nll)
        token_counts.append(len(response_ids))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "step": checkpoint_step(checkpoint),
        "checkpoint": str(checkpoint),
        "student_nll": mean(student_nlls),
        "teacher_nll": mean(teacher_nlls),
        "gap": mean(gaps),
        "examples": len(gaps),
        "tokens": sum(token_counts),
    }


def append_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fields = [
        "step",
        "checkpoint",
        "student_nll",
        "teacher_nll",
        "gap",
        "moving_improvement",
        "improvement_per_100_steps",
        "examples",
        "tokens",
    ]
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(output_dir: Path, rows: list[dict], plateau_threshold: float, plateau_patience: int):
    if not rows:
        return
    parsed = []
    for row in rows:
        parsed.append(
            {
                **row,
                "step": int(row["step"]),
                "gap": float(row["gap"]),
                "student_nll": float(row["student_nll"]),
                "teacher_nll": float(row["teacher_nll"]),
                "improvement_per_100_steps": float(row.get("improvement_per_100_steps") or 0.0),
            }
        )
    parsed.sort(key=lambda r: r["step"])
    best = min(parsed, key=lambda r: r["gap"])

    plateau = None
    if len(parsed) >= plateau_patience + 1:
        for idx in range(plateau_patience, len(parsed)):
            window = parsed[idx - plateau_patience + 1 : idx + 1]
            if all(abs(r["improvement_per_100_steps"]) < plateau_threshold for r in window):
                plateau = parsed[idx]
                break

    lines = [
        f"evaluated_checkpoints: {len(parsed)}",
        f"best_step: {best['step']}",
        f"best_gap: {best['gap']:.6f}",
        f"best_student_nll: {best['student_nll']:.6f}",
        f"teacher_nll: {best['teacher_nll']:.6f}",
        f"best_checkpoint: {best['checkpoint']}",
    ]
    if plateau:
        lines += [
            "",
            f"plateau_candidate_step: {plateau['step']}",
            f"plateau_gap: {plateau['gap']:.6f}",
            f"plateau_checkpoint: {plateau['checkpoint']}",
            f"plateau_rule: abs(improvement_per_100_steps) < {plateau_threshold} for {plateau_patience} checkpoints",
        ]
    else:
        lines += ["", "plateau_candidate_step: not_found"]

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    (output_dir / "best_checkpoint.txt").write_text(best["checkpoint"] + "\n")
    selected = plateau["checkpoint"] if plateau else best["checkpoint"]
    (output_dir / "selected_checkpoint.txt").write_text(selected + "\n")


def prune_checkpoints(args, evaluated_rows: list[dict]):
    if args.no_prune:
        return
    checkpoints = sorted(
        [p for p in Path(args.checkpoint_dir).glob("checkpoint-*") if p.is_dir()],
        key=checkpoint_step,
    )
    if not checkpoints:
        return

    latest = checkpoints[-args.keep_latest :]
    best_path = None
    if evaluated_rows:
        best = min(evaluated_rows, key=lambda r: float(r["gap"]))
        best_path = Path(best["checkpoint"]).resolve()

    keep = {p.resolve() for p in latest}
    if best_path is not None:
        keep.add(best_path)
    selected_file = Path(args.output_dir) / "selected_checkpoint.txt"
    if selected_file.exists():
        selected_text = selected_file.read_text().strip()
        if selected_text:
            keep.add(Path(selected_text).resolve())

    for ckpt in checkpoints:
        if ckpt.resolve() in keep:
            continue
        if not any(str(ckpt) == r["checkpoint"] for r in evaluated_rows):
            continue
        print(f"prune checkpoint: {ckpt}")
        shutil.rmtree(ckpt)


def evaluate_once(args):
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "sft_probe_metrics.csv"
    previous_rows = load_existing(csv_path)
    done = {int(r["step"]) for r in previous_rows}

    probe_rows = pd.read_parquet(args.probe_parquet).to_dict(orient="records")
    checkpoints = sorted(
        [p for p in Path(args.checkpoint_dir).glob("checkpoint-*") if is_checkpoint_ready(p)],
        key=checkpoint_step,
    )

    rows = previous_rows
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        if step in done:
            continue
        if args.min_age_seconds > 0 and time.time() - checkpoint.stat().st_mtime < args.min_age_seconds:
            continue
        result = eval_checkpoint(args, checkpoint, probe_rows)
        previous = rows[-1] if rows else None
        if previous:
            prev_gap = float(previous["gap"])
            prev_step = int(previous["step"])
            improvement = prev_gap - result["gap"]
            step_delta = max(result["step"] - prev_step, 1)
            result["moving_improvement"] = improvement
            result["improvement_per_100_steps"] = improvement / step_delta * 100.0
        else:
            result["moving_improvement"] = 0.0
            result["improvement_per_100_steps"] = 0.0
        append_csv(csv_path, result)
        rows.append({k: str(v) for k, v in result.items()})
        write_summary(output_dir, rows, args.plateau_threshold, args.plateau_patience)
        prune_checkpoints(args, rows)
        done.add(step)

    write_summary(output_dir, rows, args.plateau_threshold, args.plateau_patience)
    prune_checkpoints(args, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--probe-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-age-seconds", type=int, default=30)
    parser.add_argument("--keep-latest", type=int, default=1)
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument("--plateau-threshold", type=float, default=0.01)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    while True:
        evaluate_once(args)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
