# SPDX-License-Identifier: Apache-2.0

"""Distributional diagnostics for student/teacher OPD trajectories."""

from __future__ import annotations

from collections.abc import Iterable

import torch

DEFAULT_POSITION_BINS = ((1, 256), (257, 512), (513, 1024), (1025, 2048), (2049, 4096))


@torch.no_grad()
def compute_token_diagnostics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    *,
    top_k: int = 16,
    advantage_clip: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Compute paper-style top-k alignment metrics for aligned token states.

    Inputs are ``[tokens, vocabulary]`` logits and ``[tokens]`` chosen IDs.
    Returned tensors are per-token so callers can aggregate by response depth.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"Student/teacher shapes differ: {student_logits.shape} vs {teacher_logits.shape}")
    if student_logits.ndim != 2 or chosen_token_ids.shape != student_logits.shape[:1]:
        raise ValueError("Expected logits [tokens, vocab] and chosen_token_ids [tokens]")
    if not 0 < top_k <= student_logits.shape[-1]:
        raise ValueError(f"top_k must be in [1, vocab_size], got {top_k}")

    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_top_log_probs, student_top_ids = student_log_probs.topk(top_k, dim=-1)
    teacher_top_log_probs, teacher_top_ids = teacher_log_probs.topk(top_k, dim=-1)

    matches = student_top_ids.unsqueeze(-1).eq(teacher_top_ids.unsqueeze(-2))
    student_overlap_mask = matches.any(dim=-1)
    teacher_overlap_mask = matches.any(dim=-2)
    overlap_count = student_overlap_mask.sum(dim=-1)

    student_top_probs = student_top_log_probs.exp()
    teacher_top_probs = teacher_top_log_probs.exp()
    student_overlap_mass = (student_top_probs * student_overlap_mask).sum(dim=-1)
    teacher_overlap_mass = (teacher_top_probs * teacher_overlap_mask).sum(dim=-1)

    teacher_probs_aligned_to_student = (matches * teacher_top_probs.unsqueeze(-2)).sum(dim=-1)
    eps = torch.finfo(torch.float32).tiny
    student_renorm = student_top_probs * student_overlap_mask / student_overlap_mass.clamp_min(eps).unsqueeze(-1)
    teacher_renorm = (
        teacher_probs_aligned_to_student
        / teacher_overlap_mass.clamp_min(eps).unsqueeze(-1)
    )
    per_overlap_token_advantage = student_renorm * (
        teacher_renorm.clamp_min(eps).log() - student_renorm.clamp_min(eps).log()
    )
    overlap_token_advantage = per_overlap_token_advantage.sum(dim=-1) / overlap_count.clamp_min(1)

    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)

    gather_ids = chosen_token_ids.to(student_logits.device).unsqueeze(-1)
    sampled_advantage = teacher_log_probs.gather(-1, gather_ids).squeeze(-1) - student_log_probs.gather(
        -1, gather_ids
    ).squeeze(-1)

    return {
        "overlap_ratio": overlap_count.float() / top_k,
        "student_overlap_mass": student_overlap_mass,
        "teacher_overlap_mass": teacher_overlap_mass,
        "overlap_token_advantage": overlap_token_advantage,
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "entropy_gap": (teacher_entropy - student_entropy).abs(),
        "sampled_advantage": sampled_advantage,
        "sampled_advantage_positive": (sampled_advantage > 0).float(),
        "sampled_advantage_clip_fraction": (sampled_advantage.abs() > advantage_clip).float(),
    }


def summarize_token_diagnostics(
    metrics: dict[str, torch.Tensor],
    *,
    position_bins: Iterable[tuple[int, int]] = DEFAULT_POSITION_BINS,
) -> dict[str, dict[str, float]]:
    """Summarize token diagnostics overall and in one-indexed inclusive bins."""
    if not metrics:
        return {}
    length = next(iter(metrics.values())).numel()
    if any(value.numel() != length for value in metrics.values()):
        raise ValueError("All diagnostic tensors must have the same token length")

    def summarize_slice(start: int, stop: int) -> dict[str, float]:
        result = {}
        for name, values in metrics.items():
            selected = values[start:stop].float()
            result[name] = selected.mean().item() if selected.numel() else float("nan")
            if name == "sampled_advantage" and selected.numel():
                result["sampled_advantage_std"] = selected.std(unbiased=False).item()
        result["tokens"] = max(0, min(length, stop) - min(length, start))
        return result

    summary = {"overall": summarize_slice(0, length)}
    for first, last in position_bins:
        summary[f"{first}-{last}"] = summarize_slice(first - 1, min(last, length))
    return summary
