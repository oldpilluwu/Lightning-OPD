#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the official Lightning-OPD path with checkpoint-sidecar SFT metrics.

This file does not replace the repo's SFT or OPD trainers. It:
  1. prepares 5k SFT prompts and 2k OPD prompts,
  2. generates SFT data with the existing teacher curation script,
  3. launches LlamaFactory SFT using a runtime copy of the official YAML,
  4. scores saved SFT checkpoints on a fixed teacher-generated OPD probe,
  5. optionally runs the existing rollout, teacher-logprob precompute, and
     official Lightning-OPD Megatron config.

For per-step SFT metrics, set --sft-save-steps 1. That preserves the training
algorithm, but it can require very large checkpoint storage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported file format: {path}")


def normalize_prompt(row: dict[str, Any]) -> dict[str, Any] | None:
    if "prompt" in row:
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        return {"prompt": prompt, "label": str(row.get("label", "0"))}

    if "messages" in row:
        prompt = [{"role": m["role"], "content": m["content"]} for m in row["messages"] if m["role"] != "assistant"]
        return {"prompt": prompt, "label": str(row.get("label", "0"))} if prompt else None

    if "conversations" in row:
        prompt = []
        for turn in row["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                prompt.append({"role": "user", "content": content})
            elif role == "system":
                prompt.append({"role": "system", "content": content})
        return {"prompt": prompt, "label": str(row.get("label", "0"))} if prompt else None

    return None


def sample_prompt_file(src: Path, dst: Path, count: int, seed: int) -> Path:
    rows = [item for item in (normalize_prompt(row) for row in load_records(src)) if item is not None]
    if count > 0 and count < len(rows):
        rows = random.Random(seed).sample(rows, count)
    write_jsonl(dst, rows)
    return dst


def find_first_jsonl(path: Path) -> Path:
    if path.is_file() and path.suffix == ".jsonl":
        return path
    matches = sorted(path.rglob("*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No .jsonl file found under {path}")
    return matches[0]


def hf_download_dataset(dataset: str, out_dir: Path) -> None:
    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if cli is None:
        raise RuntimeError("Install huggingface_hub CLI: pip install huggingface_hub")
    run([cli, "download", "--repo-type", "dataset", dataset, "--include", "*.jsonl", "--local-dir", str(out_dir)], cwd=repo_root())


def prepare_inputs(args) -> tuple[Path, Path]:
    prompt_dir = args.run_dir / "data" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    if args.sft_prompts:
        sft_prompts = sample_prompt_file(Path(args.sft_prompts), prompt_dir / f"sft_prompts_{args.sft_samples}.jsonl", args.sft_samples, args.seed)
    else:
        sft_prompts = prompt_dir / f"openthoughts3_{args.sft_samples}.jsonl"
        if not sft_prompts.exists() or args.force:
            run(
                [
                    sys.executable,
                    "scripts/prepare_sft_prompts.py",
                    "--hf-dataset",
                    args.sft_hf_dataset,
                    "--output",
                    str(sft_prompts),
                    "--num-samples",
                    str(args.sft_samples),
                    "--seed",
                    str(args.seed),
                ],
                cwd=repo_root(),
            )

    if args.opd_prompts:
        opd_source = Path(args.opd_prompts)
    else:
        opd_download = prompt_dir / "dapo-math-17k"
        if not opd_download.exists() or args.force:
            hf_download_dataset(args.opd_hf_dataset, opd_download)
        opd_source = find_first_jsonl(opd_download)

    opd_prompts = sample_prompt_file(opd_source, prompt_dir / f"opd_prompts_{args.opd_samples}.jsonl", args.opd_samples, args.seed)
    return sft_prompts, opd_prompts


def generate_sft_data(args, sft_prompts: Path) -> Path:
    raw_dir = args.run_dir / "data" / "sft_data_raw"
    merged = args.run_dir / "data" / "sft_data" / f"openthoughts3_{args.sft_samples}_qwen3-8b.parquet"
    if merged.exists() and not args.force:
        return merged

    run(
        [
            "bash",
            "scripts/generate_sft_data.sh",
            "--num-samples",
            str(args.sft_samples),
            "--max-tokens",
            str(args.sft_gen_max_tokens),
        ],
        cwd=repo_root(),
        env={
            "TEACHER_MODEL": args.teacher_model,
            "SFT_PROMPTS": str(sft_prompts),
            "OUTPUT_DIR": str(raw_dir),
            "NUM_GPUS": str(args.curation_num_gpus),
            "TP_SIZE": str(args.teacher_tp),
        },
    )
    run([sys.executable, "data_curation/merge.py", "--input-dir", str(raw_dir), "--output", str(merged)], cwd=repo_root())
    return merged


def make_runtime_sft_config(args, sft_parquet: Path) -> Path:
    import yaml

    cfg_dir = args.run_dir / "configs" / "sft"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"openthoughts3_{args.sft_samples}_qwen3_8b_runtime"
    dataset_info = {
        dataset_name: {
            "file_name": str(sft_parquet.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        }
    }
    (cfg_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")

    base_cfg_path = repo_root() / "configs" / "sft" / "qwen3-4b-base-open-thoughts3-qwen3-8b.yaml"
    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model_name_or_path"] = args.student_model
    cfg["dataset"] = dataset_name
    cfg["max_steps"] = args.sft_max_steps
    cfg["save_steps"] = args.sft_save_steps
    cfg["save_total_limit"] = args.sft_save_total_limit
    cfg["run_name"] = f"{args.run_name}-sft"
    cfg["report_to"] = args.sft_report_to
    cfg["per_device_train_batch_size"] = args.sft_per_device_train_batch_size
    cfg["gradient_accumulation_steps"] = args.sft_gradient_accumulation_steps
    cfg["learning_rate"] = args.sft_lr

    config_path = cfg_dir / "qwen3-4b-official-sft-runtime.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


def run_sft(args, config_path: Path) -> Path:
    out_dir = args.run_dir / "checkpoints" / "qwen3-4b-base-sft-qwen3-8b"
    if out_dir.exists() and not args.force:
        print(f"[sft] Reusing existing output dir: {out_dir}", flush=True)
        return out_dir

    run(
        [
            "torchrun",
            "--nnodes",
            str(args.sft_num_nodes),
            "--nproc_per_node",
            str(args.sft_num_gpus),
            "--rdzv_id",
            str(random.randint(1, 1_000_000)),
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            f"{args.master_addr}:{args.master_port}",
            "-m",
            "llamafactory.cli.train",
            str(config_path),
            f"dataset_dir={config_path.parent}",
            f"output_dir={out_dir}",
        ],
        cwd=repo_root(),
    )
    return out_dir


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else 10**18


def list_checkpoints(sft_output_dir: Path) -> list[Path]:
    return sorted((p for p in sft_output_dir.glob("checkpoint-*") if p.is_dir()), key=checkpoint_step)


def score_response_logps(model, input_ids, prompt_len: int) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, :-1].float()
        targets = input_ids[0, 1:]
        response_len = input_ids.shape[1] - prompt_len
        start = max(prompt_len - 1, 0)
        logps = torch.log_softmax(logits[start : start + response_len], dim=-1)
        token_logps = logps.gather(-1, targets[start : start + response_len].unsqueeze(-1)).squeeze(-1)
    return token_logps.detach().cpu().tolist()


def build_or_load_probe(args, tokenizer) -> list[dict[str, Any]]:
    probe_path = args.run_dir / "metrics" / "opd_teacher_probe.jsonl"
    if probe_path.exists() and not args.force_probe:
        return read_jsonl(probe_path)

    import torch
    from transformers import AutoModelForCausalLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.opd_prompts_sampled)[: args.probe_samples]

    print(f"[probe] Building fixed OPD probe with teacher {args.teacher_model}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    teacher.eval()

    probe = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            full = teacher.generate(
                input_ids=input_ids,
                max_new_tokens=args.probe_max_new_tokens,
                do_sample=True,
                temperature=args.probe_temperature,
                top_p=args.probe_top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0].tolist()
        response_ids = full[len(prompt_ids) :]
        if not response_ids:
            continue
        full_ids = prompt_ids + response_ids
        teacher_logps = score_response_logps(teacher, torch.tensor([full_ids], dtype=torch.long, device="cuda"), len(prompt_ids))
        probe.append({"prompt_ids": prompt_ids, "response_ids": response_ids, "teacher_logps": teacher_logps})

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    write_jsonl(probe_path, probe)
    return probe


def append_metric_row(jsonl_path: Path, csv_path: Path, row: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_sft_checkpoints(args, sft_output_dir: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoints = list_checkpoints(sft_output_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs found under {sft_output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe = build_or_load_probe(args, tokenizer)
    if not probe:
        raise RuntimeError("OPD probe is empty.")

    metrics_jsonl = args.run_dir / "metrics" / "sft_checkpoint_opd_probe_metrics.jsonl"
    metrics_csv = args.run_dir / "metrics" / "sft_checkpoint_opd_probe_metrics.csv"
    seen_steps = set()
    if metrics_jsonl.exists() and not args.force_metrics:
        for row in read_jsonl(metrics_jsonl):
            seen_steps.add(row["step"])

    prev_student_logps: list[list[float]] | None = None
    for ckpt in checkpoints:
        step = checkpoint_step(ckpt)
        should_write = step not in seen_steps

        print(f"[metrics] Scoring SFT checkpoint step={step}: {ckpt}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        model.eval()

        teacher_nll_sum = 0.0
        student_nll_sum = 0.0
        token_count = 0
        current_student_logps = []
        for item in probe:
            full_ids = item["prompt_ids"] + item["response_ids"]
            logps = score_response_logps(model, torch.tensor([full_ids], dtype=torch.long, device="cuda"), len(item["prompt_ids"]))
            current_student_logps.append(logps)
            teacher_nll_sum += -sum(item["teacher_logps"])
            student_nll_sum += -sum(logps)
            token_count += len(logps)

        drift_signed = 0.0
        drift_abs = 0.0
        drift_tokens = 0
        if prev_student_logps is not None:
            for prev, cur in zip(prev_student_logps, current_student_logps, strict=False):
                for p, c in zip(prev, cur, strict=False):
                    drift_signed += p - c
                    drift_abs += abs(p - c)
                    drift_tokens += 1
        prev_student_logps = current_student_logps

        token_count = max(token_count, 1)
        drift_tokens = max(drift_tokens, 1)
        teacher_nll = teacher_nll_sum / token_count
        student_nll = student_nll_sum / token_count
        row = {
            "step": step,
            "checkpoint": str(ckpt),
            "probe/tokens": token_count,
            "probe/teacher_nll": teacher_nll,
            "probe/student_nll": student_nll,
            "probe/kl_mc_teacher_to_student": student_nll - teacher_nll,
            "probe/policy_drift_prev_to_current_mc": drift_signed / drift_tokens,
            "probe/policy_drift_abs_logprob_delta": drift_abs / drift_tokens,
        }
        if should_write:
            append_metric_row(metrics_jsonl, metrics_csv, row)
            print("[metrics] " + json.dumps(row, sort_keys=True), flush=True)
        else:
            print(f"[metrics] Context only, row already exists for step={step}", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()


def collect_rollouts(args, sft_checkpoint: Path, opd_prompts: Path) -> Path:
    raw_dir = args.run_dir / "data" / "rollouts_raw"
    merged = args.run_dir / "data" / "rollouts" / f"dapo_{args.opd_samples}_qwen3-4b-sft-rollouts.parquet"
    if merged.exists() and not args.force:
        return merged
    run(
        [
            "bash",
            "scripts/collect_rollouts.sh",
            "--num-samples",
            str(args.opd_samples),
            "--max-tokens",
            str(args.opd_rollout_max_tokens),
        ],
        cwd=repo_root(),
        env={
            "SFT_CHECKPOINT": str(sft_checkpoint),
            "OPD_PROMPTS": str(opd_prompts),
            "OUTPUT_DIR": str(raw_dir),
            "NUM_GPUS": str(args.curation_num_gpus),
            "TP_SIZE": str(args.student_tp),
        },
    )
    run([sys.executable, "data_curation/merge.py", "--input-dir", str(raw_dir), "--output", str(merged)], cwd=repo_root())
    return merged


def start_teacher_server(args) -> subprocess.Popen:
    log_file = args.run_dir / "logs" / "sglang_teacher_8b.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.teacher_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.teacher_port),
        "--tp",
        str(args.teacher_tp),
        "--chunked-prefill-size",
        "4096",
        "--mem-fraction-static",
        str(args.teacher_mem_fraction),
        "--context-length",
        str(args.teacher_context_length),
    ]
    print("\n$ " + " ".join(cmd), flush=True)
    handle = log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(repo_root()), stdout=handle, stderr=subprocess.STDOUT)

    import urllib.request

    health_url = f"http://127.0.0.1:{args.teacher_port}/health_generate"
    for _ in range(args.teacher_start_timeout_s // 5):
        if proc.poll() is not None:
            raise RuntimeError(f"Teacher server exited early; see {log_file}")
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f"[teacher] Ready at {health_url}", flush=True)
            return proc
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"Teacher server did not become healthy; see {log_file}")


def precompute_lightning_opd(args, sft_checkpoint: Path, rollout_parquet: Path) -> Path:
    out_dir = args.run_dir / "data" / "lightning_opd"
    final = out_dir / f"{rollout_parquet.stem}-lightning-opd-precomputed.parquet"
    if final.exists() and not args.force:
        return final

    proc = start_teacher_server(args)
    try:
        run(
            [
                sys.executable,
                "data_curation/prepare_lightning_opd.py",
                "--tokenizer-path",
                str(sft_checkpoint),
                "--input-parquet",
                str(rollout_parquet),
                "--output-dir",
                str(out_dir),
                "--max-response-len",
                str(args.opd_rollout_max_tokens),
                "--compute-teacher-logprobs",
                "--teacher-url",
                f"http://127.0.0.1:{args.teacher_port}/generate",
                "--concurrency",
                str(args.teacher_logprob_concurrency),
            ],
            cwd=repo_root(),
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return final


def run_official_lightning_opd(args, sft_checkpoint: Path, lightning_data: Path) -> None:
    run(
        [sys.executable, "configs/lightning_opd/qwen3-4b-lightning-opd.py"],
        cwd=repo_root(),
        env={
            "SFT_CHECKPOINT": str(sft_checkpoint),
            "LIGHTNING_OPD_DATA": str(lightning_data),
            **({"WANDB_KEY": args.wandb_key} if args.wandb_key else {}),
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official Lightning-OPD run plus SFT checkpoint OPD-probe metrics.")
    p.add_argument("--run-name", default="official_qwen3_4b_student_qwen3_8b_teacher")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--student-model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-8B")
    p.add_argument("--sft-hf-dataset", default="open-thoughts/OpenThoughts3-1.2M")
    p.add_argument("--opd-hf-dataset", default="zhuzilin/dapo-math-17k")
    p.add_argument("--sft-prompts")
    p.add_argument("--opd-prompts")
    p.add_argument("--sft-samples", type=int, default=5000)
    p.add_argument("--opd-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")

    p.add_argument("--sft-num-nodes", type=int, default=1)
    p.add_argument("--sft-num-gpus", type=int, default=1)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--sft-max-steps", type=int, default=3000)
    p.add_argument("--sft-save-steps", type=int, default=100)
    p.add_argument("--sft-save-total-limit", type=int, default=10)
    p.add_argument("--sft-per-device-train-batch-size", type=int, default=4)
    p.add_argument("--sft-gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--sft-lr", type=float, default=8e-5)
    p.add_argument("--sft-report-to", default="wandb")
    p.add_argument("--sft-gen-max-tokens", type=int, default=16384)

    p.add_argument("--probe-samples", type=int, default=8)
    p.add_argument("--probe-max-new-tokens", type=int, default=512)
    p.add_argument("--probe-temperature", type=float, default=0.7)
    p.add_argument("--probe-top-p", type=float, default=0.9)
    p.add_argument("--force-probe", action="store_true")
    p.add_argument("--force-metrics", action="store_true")
    p.add_argument("--attn-implementation", default="flash_attention_2")

    p.add_argument("--curation-num-gpus", type=int, default=1)
    p.add_argument("--teacher-tp", type=int, default=1)
    p.add_argument("--student-tp", type=int, default=1)
    p.add_argument("--teacher-port", type=int, default=13141)
    p.add_argument("--teacher-mem-fraction", type=float, default=0.72)
    p.add_argument("--teacher-context-length", type=int, default=8192)
    p.add_argument("--teacher-start-timeout-s", type=int, default=900)
    p.add_argument("--teacher-logprob-concurrency", type=int, default=16)
    p.add_argument("--opd-rollout-max-tokens", type=int, default=4096)

    p.add_argument("--metrics-only", type=Path, help="Only score an existing SFT output directory.")
    p.add_argument("--stop-after-sft-metrics", action="store_true")
    p.add_argument("--skip-lightning-opd", action="store_true")
    p.add_argument("--wandb-key", default=os.environ.get("WANDB_KEY"))
    args = p.parse_args()

    if args.run_dir is None:
        args.run_dir = repo_root() / "runs" / args.run_name
    args.run_dir = args.run_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    if args.metrics_only:
        if not args.opd_prompts:
            raise ValueError("--opd-prompts is required with --metrics-only")
        args.opd_prompts_sampled = sample_prompt_file(Path(args.opd_prompts), args.run_dir / "data" / "prompts" / f"opd_prompts_{args.opd_samples}.jsonl", args.opd_samples, args.seed)
        evaluate_sft_checkpoints(args, args.metrics_only)
        return

    sft_prompts, opd_prompts = prepare_inputs(args)
    args.opd_prompts_sampled = opd_prompts
    sft_parquet = generate_sft_data(args, sft_prompts)
    sft_config = make_runtime_sft_config(args, sft_parquet)
    sft_output_dir = run_sft(args, sft_config)

    evaluate_sft_checkpoints(args, sft_output_dir)
    if args.stop_after_sft_metrics:
        print(f"[done] SFT metrics: {args.run_dir / 'metrics' / 'sft_checkpoint_opd_probe_metrics.csv'}")
        return

    rollout_parquet = collect_rollouts(args, sft_output_dir, opd_prompts)
    lightning_data = precompute_lightning_opd(args, sft_output_dir, rollout_parquet)

    if args.skip_lightning_opd:
        print(f"[done] Lightning OPD data: {lightning_data}")
        return

    run_official_lightning_opd(args, sft_output_dir, lightning_data)


if __name__ == "__main__":
    main()
