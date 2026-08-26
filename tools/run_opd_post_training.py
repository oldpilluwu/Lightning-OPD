#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Convert, evaluate, diagnose, and report a completed Qwen3 OPD run.

Every result is appended to JSONL before moving to the next item, making the
pipeline resumable without touching training checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.opd_diagnostics import compute_token_diagnostics, summarize_token_diagnostics  # noqa: E402

_SCORER_SPEC = importlib.util.spec_from_file_location(
    "opd_math_dapo_utils", REPO_ROOT / "slime" / "rollout" / "rm_hub" / "math_dapo_utils.py"
)
_SCORER_MODULE = importlib.util.module_from_spec(_SCORER_SPEC)
_SCORER_SPEC.loader.exec_module(_SCORER_MODULE)
compute_score = _SCORER_MODULE.compute_score

CHECKPOINT_STEPS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 75, 100, 125, 150)
FULL_BENCHMARK_STEPS = (0, 25, 50, 75, 100, 125, 150)
ENDPOINT_STEPS = (0, 150)
MATH_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response must be "
    "exactly `Answer: <answer>`.\n\n"
)


def parse_steps(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def converted_model_path(step: int, origin_hf: Path, hf_root: Path) -> Path:
    return origin_hf if step == 0 else hf_root / f"step_{step:06d}"


def convert_checkpoints(args: argparse.Namespace) -> None:
    for step in args.steps:
        source = args.checkpoint_root / f"iter_{step:07d}"
        destination = converted_model_path(step, args.origin_hf, args.hf_root)
        complete = destination / ".conversion_complete"
        if complete.exists():
            print(f"[convert] step {step}: already complete")
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing Megatron checkpoint for step {step}: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "MEGATRON_CKPT_DIR": str(source),
            "HF_OUTPUT_DIR": str(destination),
            "ORIGIN_HF_DIR": str(args.origin_hf),
        }
        subprocess.run(
            ["bash", "scripts/convert_megatron_to_hf.sh"], check=True, env=env, cwd=REPO_ROOT
        )
        source_manifest = source / "manifest.json"
        if source_manifest.exists():
            (destination / "opd_manifest.json").write_text(
                source_manifest.read_text(encoding="utf-8"), encoding="utf-8"
            )
        complete.write_text(f"step={step}\n", encoding="utf-8")


def _format_prompt(tokenizer, problem: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": MATH_INSTRUCTION + problem}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _load_model(path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def _score(response: str, label: str) -> tuple[bool, bool, str | None]:
    result = compute_score(response, label)
    prediction = result.get("pred")
    valid = prediction is not None and str(prediction).strip() != ""
    return bool(result.get("acc")), valid, None if prediction is None else str(prediction)


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device: str,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    prompts = [_format_prompt(tokenizer, row["prompt"]) for row in rows]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    do_sample = temperature is not None and temperature > 0
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_args.update(temperature=temperature, top_p=top_p, top_k=top_k)
    generated = model.generate(**encoded, **generation_args)
    input_width = encoded["input_ids"].shape[1]
    results = []
    for row, token_ids in zip(rows, generated[:, input_width:], strict=True):
        response_ids = token_ids.tolist()
        if tokenizer.eos_token_id in response_ids:
            response_ids = response_ids[: response_ids.index(tokenizer.eos_token_id) + 1]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        correct, valid, prediction = _score(response, row["label"])
        results.append(
            {
                "response": response,
                "prediction": prediction,
                "correct": correct,
                "valid_answer": valid,
                "response_tokens": len(response_ids),
                "truncated": len(response_ids) >= max_new_tokens,
            }
        )
    return results


def _existing_outcome_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    return {
        (row["step"], row["suite"], row["dataset"], row["problem_id"], row["generation_index"])
        for row in read_jsonl(path)
    }


def _run_outcome_suite(
    *,
    model,
    tokenizer,
    step: int,
    suite: str,
    datasets: dict[str, list[dict[str, Any]]],
    generations: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    args: argparse.Namespace,
    existing: set[tuple],
) -> None:
    for dataset_name, rows in datasets.items():
        for generation_index in range(generations):
            pending = [
                row
                for row in rows
                if (step, suite, dataset_name, row["id"], generation_index) not in existing
            ]
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                outputs = generate_batch(
                    model,
                    tokenizer,
                    batch,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=args.seed + generation_index,
                )
                for source, output in zip(batch, outputs, strict=True):
                    result = {
                        "step": step,
                        "suite": suite,
                        "dataset": dataset_name,
                        "problem_id": source["id"],
                        "generation_index": generation_index,
                        "label": source["label"],
                        "training_overlap_exact": source.get("training_overlap_exact"),
                        "training_overlap_semantic": source.get("training_overlap_semantic"),
                        "training_overlap_cosine": source.get("training_overlap_cosine"),
                        **output,
                    }
                    append_jsonl(args.output_dir / "outcomes.jsonl", result)
                    existing.add((step, suite, dataset_name, source["id"], generation_index))


def run_outcome_benchmarks(args: argparse.Namespace) -> None:
    data = {
        name: read_jsonl(args.data_dir / f"{name}.jsonl")
        for name in ("math500", "aime24", "aime25", "amc23")
    }
    diagnostic_data = {
        "math500_diagnostic": read_jsonl(args.data_dir / "math500_diagnostic.jsonl"),
        "aime24": data["aime24"],
        "aime25": data["aime25"],
        "amc23": data["amc23"],
    }
    existing = _existing_outcome_keys(args.output_dir / "outcomes.jsonl")

    for step in (0, *args.steps):
        model_path = converted_model_path(step, args.origin_hf, args.hf_root)
        print(f"[outcomes] loading step {step}: {model_path}")
        model, tokenizer = _load_model(model_path, args.device)
        _run_outcome_suite(
            model=model,
            tokenizer=tokenizer,
            step=step,
            suite="diagnostic_greedy",
            datasets=diagnostic_data,
            generations=1,
            temperature=None,
            top_p=None,
            top_k=None,
            args=args,
            existing=existing,
        )
        if step in args.full_benchmark_steps:
            _run_outcome_suite(
                model=model,
                tokenizer=tokenizer,
                step=step,
                suite="full_sampled_n1",
                datasets=data,
                generations=1,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                args=args,
                existing=existing,
            )
        if step in ENDPOINT_STEPS:
            _run_outcome_suite(
                model=model,
                tokenizer=tokenizer,
                step=step,
                suite="full_sampled_n4",
                datasets=data,
                generations=4,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                args=args,
                existing=existing,
            )
        del model
        torch.cuda.empty_cache()


def _tokenizer_signature(tokenizer) -> dict[str, Any]:
    return {
        "vocab_size": len(tokenizer),
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
    }


def _validate_tokenizers(student_tokenizer, teacher_tokenizer) -> None:
    student_signature = _tokenizer_signature(student_tokenizer)
    teacher_signature = _tokenizer_signature(teacher_tokenizer)
    if student_signature != teacher_signature:
        raise ValueError(f"Tokenizer special-ID mismatch: {student_signature} != {teacher_signature}")
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher token-to-ID vocabularies differ")
    fixture = "Verify tokenizer compatibility: $1+1=2$."
    student_ids = student_tokenizer(_format_prompt(student_tokenizer, fixture), add_special_tokens=False).input_ids
    teacher_ids = teacher_tokenizer(_format_prompt(teacher_tokenizer, fixture), add_special_tokens=False).input_ids
    if student_ids != teacher_ids:
        raise ValueError("Student and teacher non-thinking chat templates tokenize differently")


@torch.inference_mode()
def _diagnose_one(
    student,
    teacher,
    tokenizer,
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = _format_prompt(tokenizer, row["prompt"])
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    torch.manual_seed(args.seed)
    generated = student.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_length = encoded["input_ids"].shape[1]
    response_ids = generated[0, prompt_length:]
    if tokenizer.eos_token_id in response_ids.tolist():
        eos_position = response_ids.tolist().index(tokenizer.eos_token_id) + 1
        response_ids = response_ids[:eos_position]
        generated = generated[:, : prompt_length + eos_position]

    student_logits = student(generated, use_cache=False).logits[0, prompt_length - 1 : -1]
    teacher_logits = teacher(generated, use_cache=False).logits[0, prompt_length - 1 : -1]
    metrics_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, response_ids.numel(), args.logit_chunk_size):
        stop = min(response_ids.numel(), start + args.logit_chunk_size)
        part = compute_token_diagnostics(
            student_logits[start:stop],
            teacher_logits[start:stop],
            response_ids[start:stop],
            top_k=16,
            advantage_clip=10.0,
        )
        for name, values in part.items():
            metrics_parts[name].append(values.cpu())
    del student_logits, teacher_logits
    metrics = {name: torch.cat(parts) for name, parts in metrics_parts.items()}
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    correct, valid, prediction = _score(response, row["label"])
    return {
        "problem_id": row["id"],
        "source": row["source"],
        "label": row["label"],
        "prediction": prediction,
        "correct": correct,
        "valid_answer": valid,
        "response_tokens": response_ids.numel(),
        "truncated": response_ids.numel() >= args.max_new_tokens,
        "metrics": summarize_token_diagnostics(metrics),
    }


def run_distributional_diagnostics(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.data_dir / "dapo_diagnostic.jsonl") + read_jsonl(
        args.data_dir / "math500_diagnostic.jsonl"
    )
    result_path = args.output_dir / "distributional_diagnostics.jsonl"
    existing = set()
    if result_path.exists():
        existing = {(row["step"], row["source"], row["problem_id"]) for row in read_jsonl(result_path)}

    teacher, teacher_tokenizer = _load_model(args.teacher_model, args.device)
    for step in (0, *args.steps):
        student_path = converted_model_path(step, args.origin_hf, args.hf_root)
        student, tokenizer = _load_model(student_path, args.device)
        _validate_tokenizers(tokenizer, teacher_tokenizer)
        for row in rows:
            key = (step, row["source"], row["id"])
            if key in existing:
                continue
            result = {"step": step, **_diagnose_one(student, teacher, tokenizer, row, args=args)}
            append_jsonl(result_path, result)
            existing.add(key)
        del student
        torch.cuda.empty_cache()
    del teacher
    torch.cuda.empty_cache()


def _bootstrap_ci(problem_scores: dict[str, list[float]], seed: int, samples: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    ids = sorted(problem_scores)
    means = []
    for _ in range(samples):
        chosen = [rng.choice(ids) for _ in ids]
        means.append(float(np.mean([np.mean(problem_scores[item]) for item in chosen])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_report(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = read_jsonl(args.output_dir / "outcomes.jsonl")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in outcomes:
        grouped[(row["step"], row["suite"], row["dataset"])].append(row)

    outcome_summary = []
    for (step, suite, dataset), rows in sorted(grouped.items()):
        by_problem: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_problem[row["problem_id"]].append(float(row["correct"]))
        accuracy = float(np.mean([np.mean(scores) for scores in by_problem.values()]))
        low, high = _bootstrap_ci(by_problem, args.seed) if suite == "full_sampled_n4" else (math.nan, math.nan)
        outcome_summary.append(
            {
                "step": step,
                "suite": suite,
                "dataset": dataset,
                "problems": len(by_problem),
                "generations": len(rows),
                "accuracy": accuracy,
                "ci95_low": low,
                "ci95_high": high,
                "valid_answer_rate": float(np.mean([row["valid_answer"] for row in rows])),
                "mean_response_tokens": float(np.mean([row["response_tokens"] for row in rows])),
                "truncation_rate": float(np.mean([row["truncated"] for row in rows])),
                "overlap_free_accuracy": float(
                    np.mean(
                        [
                            row["correct"]
                            for row in rows
                            if not row.get("training_overlap_semantic", False)
                        ]
                    )
                )
                if any(not row.get("training_overlap_semantic", False) for row in rows)
                else math.nan,
            }
        )

    diagnostics = read_jsonl(args.output_dir / "distributional_diagnostics.jsonl")
    diagnostic_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in diagnostics:
        for position_bin, metrics in row["metrics"].items():
            diagnostic_groups[(row["step"], row["source"], position_bin)].append(metrics)
    diagnostic_summary = []
    for (step, source, position_bin), metrics_rows in sorted(diagnostic_groups.items()):
        metric_names = [name for name in metrics_rows[0] if name != "tokens"]
        diagnostic_summary.append(
            {
                "step": step,
                "source": source,
                "position_bin": position_bin,
                "prompts": len(metrics_rows),
                **{
                    name: float(np.nanmean([row[name] for row in metrics_rows]))
                    for name in metric_names
                },
            }
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "outcome_summary.csv", outcome_summary)
    write_csv(args.output_dir / "diagnostic_summary.csv", diagnostic_summary)

    report_lines = [
        "# Qwen3-1.7B → Qwen3-4B OPD report",
        "",
        "## Outcome benchmarks",
        "",
        "| Step | Suite | Dataset | Accuracy | Overlap-free | Valid | Mean tokens | Truncated | 95% CI |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in outcome_summary:
        ci = (
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
            if not math.isnan(row["ci95_low"])
            else "—"
        )
        report_lines.append(
            f"| {row['step']} | {row['suite']} | {row['dataset']} | {row['accuracy']:.3f} | "
            f"{row['overlap_free_accuracy']:.3f} | {row['valid_answer_rate']:.3f} | {row['mean_response_tokens']:.1f} | "
            f"{row['truncation_rate']:.3f} | {ci} |"
        )
    report_lines.extend(
        [
            "",
            "## Distributional diagnostics (overall)",
            "",
            "| Step | Source | Overlap | Student mass | Teacher mass | Overlap advantage | Entropy gap | Sampled advantage | Clip fraction |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic_summary:
        if row["position_bin"] != "overall":
            continue
        report_lines.append(
            f"| {row['step']} | {row['source']} | {row['overlap_ratio']:.3f} | "
            f"{row['student_overlap_mass']:.3f} | {row['teacher_overlap_mass']:.3f} | "
            f"{row['overlap_token_advantage']:.4f} | {row['entropy_gap']:.3f} | "
            f"{row['sampled_advantage']:.3f} | {row['sampled_advantage_clip_fraction']:.3f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _build_plots(args.output_dir, outcomes, outcome_summary, diagnostic_summary)


def _build_plots(
    output_dir: Path,
    raw_outcomes: list[dict],
    outcomes: list[dict],
    diagnostics: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    diagnostic_outcomes = [row for row in outcomes if row["suite"] == "diagnostic_greedy"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in sorted({row["dataset"] for row in diagnostic_outcomes}):
        rows = sorted((row for row in diagnostic_outcomes if row["dataset"] == dataset), key=lambda row: row["step"])
        ax.plot([row["step"] for row in rows], [row["accuracy"] for row in rows], marker="o", label=dataset)
    ax.set(xlabel="Completed OPD updates", ylabel="Accuracy", title="Checkpoint diagnostic accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_accuracy.png", dpi=180)
    plt.close(fig)

    overall = [row for row in diagnostics if row["position_bin"] == "overall"]
    metric_names = ("overlap_ratio", "entropy_gap", "sampled_advantage")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, metric_names, strict=True):
        for source in sorted({row["source"] for row in overall}):
            rows = sorted((row for row in overall if row["source"] == source), key=lambda row: row["step"])
            ax.plot([row["step"] for row in rows], [row[metric] for row in rows], marker="o", label=source)
        ax.set(xlabel="Completed OPD updates", ylabel=metric.replace("_", " "))
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_dynamics.png", dpi=180)
    plt.close(fig)

    length_steps = sorted(
        {row["step"] for row in raw_outcomes if row["suite"] == "diagnostic_greedy"}
    )
    length_values = [
        [
            row["response_tokens"]
            for row in raw_outcomes
            if row["suite"] == "diagnostic_greedy" and row["step"] == step
        ]
        for step in length_steps
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(length_values, tick_labels=length_steps, showfliers=False)
    ax.set(
        xlabel="Completed OPD updates",
        ylabel="Generated response tokens",
        title="Checkpoint response-length distributions",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "response_length_distributions.png", dpi=180)
    plt.close(fig)

    bins = ("1-256", "257-512", "513-1024", "1025-2048", "2049-4096")
    steps = sorted({row["step"] for row in diagnostics})
    for metric, title, filename in (
        ("entropy_gap", "Entropy-gap heatmap", "entropy_gap_heatmap.png"),
        ("overlap_ratio", "Top-16 overlap heatmap", "overlap_ratio_heatmap.png"),
    ):
        matrix = np.full((len(steps), len(bins)), np.nan)
        for i, step in enumerate(steps):
            for j, position_bin in enumerate(bins):
                values = [
                    row[metric]
                    for row in diagnostics
                    if row["step"] == step and row["position_bin"] == position_bin
                ]
                if values:
                    matrix[i, j] = np.nanmean(values)
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(bins)), bins, rotation=30, ha="right")
        ax.set_yticks(range(len(steps)), steps)
        ax.set(xlabel="Response-token position", ylabel="Completed OPD updates", title=title)
        fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("convert", "outcomes", "diagnostics", "report", "all"))
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--origin-hf", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, default=Path("/root/models/Qwen3-4B"))
    parser.add_argument("--data-dir", type=Path, default=Path("/root/datasets/qwen3-1.7b-opd"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3-1.7b-a6000-opd"))
    parser.add_argument("--steps", type=parse_steps, default=CHECKPOINT_STEPS)
    parser.add_argument("--full-benchmark-steps", type=parse_steps, default=FULL_BENCHMARK_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in ("convert", "all"):
        convert_checkpoints(args)
    if args.command in ("outcomes", "all"):
        run_outcome_benchmarks(args)
    if args.command in ("diagnostics", "all"):
        run_distributional_diagnostics(args)
    if args.command in ("report", "all"):
        build_report(args)


if __name__ == "__main__":
    main()
