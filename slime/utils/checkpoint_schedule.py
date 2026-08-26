# SPDX-License-Identifier: Apache-2.0

"""Utilities for explicit, completed-update checkpoint schedules."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path


def parse_step_list(value: str | Iterable[int] | None) -> tuple[int, ...]:
    """Parse a comma-separated checkpoint schedule into sorted unique steps."""
    if value is None or value == "":
        return ()
    if not isinstance(value, str):
        steps = [int(step) for step in value]
    else:
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(step <= 0 for step in steps):
        raise ValueError(f"Checkpoint steps must be positive completed-update counts, got {steps}")
    return tuple(sorted(set(steps)))


def scheduled_checkpoint(
    completed_step: int,
    checkpoint_steps: Iterable[int],
    optimizer_checkpoint_steps: Iterable[int],
) -> tuple[bool, bool]:
    """Return ``(should_save, include_optimizer)`` for a completed update."""
    checkpoint_steps = set(checkpoint_steps)
    optimizer_checkpoint_steps = set(optimizer_checkpoint_steps)
    return completed_step in checkpoint_steps, completed_step in optimizer_checkpoint_steps


def validate_checkpoint_schedule(args: Namespace) -> None:
    checkpoint_steps = set(getattr(args, "checkpoint_steps", ()) or ())
    optimizer_steps = set(getattr(args, "optimizer_checkpoint_steps", ()) or ())
    if not checkpoint_steps and not optimizer_steps:
        return
    if not checkpoint_steps:
        raise ValueError("--optimizer-checkpoint-steps requires --checkpoint-steps")
    if not optimizer_steps.issubset(checkpoint_steps):
        missing = sorted(optimizer_steps - checkpoint_steps)
        raise ValueError(f"Optimizer checkpoint steps must also be checkpoint steps; missing {missing}")
    if args.save is None:
        raise ValueError("--save is required when --checkpoint-steps is set")
    if args.save_interval is not None:
        raise ValueError("Use either --save-interval or --checkpoint-steps, not both")
    if args.async_save:
        raise ValueError("Selective weight-only checkpoints do not support --async-save")
    if args.num_rollout is not None and max(checkpoint_steps) > args.num_rollout:
        raise ValueError(
            f"Checkpoint step {max(checkpoint_steps)} exceeds --num-rollout {args.num_rollout}"
        )


def checkpoint_directory(save_root: str | os.PathLike[str], completed_step: int) -> Path:
    return Path(save_root) / f"iter_{completed_step:07d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_checkpoint_manifest(
    checkpoint_dir: Path,
    *,
    args: Namespace,
    completed_step: int,
    include_optimizer: bool,
    trained_response_tokens: int,
    trained_total_tokens: int,
) -> Path:
    """Write a reproducibility manifest after all checkpoint shards are durable."""
    files = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(checkpoint_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    manifest = {
        "schema_version": 1,
        "completed_updates": completed_step,
        "checkpoint_kind": "resumable" if include_optimizer else "weights_only",
        "optimizer_state": include_optimizer,
        "scheduler_state": include_optimizer,
        "rng_state": include_optimizer,
        "git_commit": _git_commit(),
        "created_unix": time.time(),
        "source_hf_checkpoint": getattr(args, "hf_checkpoint", None),
        "student_model_id": getattr(args, "opd_student_model_id", None),
        "student_model_revision": getattr(args, "opd_student_model_revision", None),
        "teacher_model_id": getattr(args, "opd_teacher_model_id", None),
        "teacher_model_revision": getattr(args, "opd_teacher_model_revision", None),
        "seed": getattr(args, "seed", None),
        "rollout_seed": getattr(args, "rollout_seed", None),
        "rollout_batch_size": getattr(args, "rollout_batch_size", None),
        "global_batch_size": getattr(args, "global_batch_size", None),
        "samples_seen": completed_step
        * getattr(args, "rollout_batch_size", 0)
        * getattr(args, "n_samples_per_prompt", 1),
        "data_cursor": completed_step * getattr(args, "rollout_batch_size", 0),
        "trained_response_tokens": int(trained_response_tokens),
        "trained_total_tokens": int(trained_total_tokens),
        "hyperparameters": {
            "lr": getattr(args, "lr", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "adam_beta1": getattr(args, "adam_beta1", None),
            "adam_beta2": getattr(args, "adam_beta2", None),
            "rollout_temperature": getattr(args, "rollout_temperature", None),
            "rollout_top_p": getattr(args, "rollout_top_p", None),
            "rollout_top_k": getattr(args, "rollout_top_k", None),
            "rollout_max_prompt_len": getattr(args, "rollout_max_prompt_len", None),
            "rollout_max_response_len": getattr(args, "rollout_max_response_len", None),
            "opd_advantage_clip": getattr(args, "opd_advantage_clip", None),
            "enable_thinking": (getattr(args, "apply_chat_template_kwargs", None) or {}).get(
                "enable_thinking"
            ),
        },
        "files": files,
    }
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
