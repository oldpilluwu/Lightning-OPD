#!/usr/bin/env python3

import argparse
import ast
import csv
import re
from pathlib import Path


ROLL_RE = re.compile(r"rollout\s+(\d+):\s+(\{.*\})")
STEP_RE = re.compile(r"step\s+(\d+):\s+(\{.*\})")


def parse_dict(text: str) -> dict:
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def read_metrics(log_dir: Path):
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            for kind, regex in (("rollout", ROLL_RE), ("train", STEP_RE)):
                match = regex.search(line)
                if not match:
                    continue
                step = int(match.group(1))
                metrics = parse_dict(match.group(2))
                if metrics:
                    row = {"source": path.name, "kind": kind, "step": step}
                    row.update(metrics)
                    rows.append(row)
    return rows


def write_csv(rows: list[dict], output: Path):
    keys = ["source", "kind", "step"]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict, key: str):
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize(rows: list[dict]) -> str:
    rollout_rows = [r for r in rows if r["kind"] == "rollout"]
    train_rows = [r for r in rows if r["kind"] == "train"]

    lines = []
    lines.append(f"rollout rows: {len(rollout_rows)}")
    lines.append(f"train rows:   {len(train_rows)}")

    if rollout_rows:
        first = rollout_rows[0]
        last = rollout_rows[-1]
        lines.append("")
        lines.append("Rollout drift signals:")
        for key in ["rollout/log_probs", "rollout/ref_log_probs", "rollout/advantages", "rollout/returns"]:
            a = numeric(first, key)
            b = numeric(last, key)
            if a is not None and b is not None:
                lines.append(f"  {key}: first={a:.6f} last={b:.6f} delta={b-a:.6f}")

        cur = numeric(last, "rollout/log_probs")
        ref = numeric(last, "rollout/ref_log_probs")
        if cur is not None and ref is not None:
            lines.append(f"  final current_minus_ref_logprob={cur-ref:.6f}")

    if train_rows:
        first = train_rows[0]
        last = train_rows[-1]
        lines.append("")
        lines.append("Train drift/update signals:")
        for key in [
            "train/loss",
            "train/ppo_kl",
            "train/kl_loss",
            "train/entropy",
            "train/train_rollout_logprob_abs_diff",
            "train/grad_norm",
        ]:
            a = numeric(first, key)
            b = numeric(last, key)
            if a is not None and b is not None:
                lines.append(f"  {key}: first={a:.6f} last={b:.6f} delta={b-a:.6f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  rollout/advantages is roughly teacher_logprob - current_policy_logprob on response tokens.")
    lines.append("  rollout/log_probs moving away from rollout/ref_log_probs indicates drift from the SFT reference.")
    lines.append("  train/ppo_kl and train/kl_loss show per-update movement; persistent growth means stronger drift.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(log_dir)
    write_csv(rows, output_dir / "metrics.csv")
    summary = summarize(rows)
    (output_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
