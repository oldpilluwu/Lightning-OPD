import torch

from slime.utils.opd_diagnostics import compute_token_diagnostics, summarize_token_diagnostics


def test_identical_distributions_have_perfect_alignment():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    chosen = torch.tensor([0, 3])
    metrics = compute_token_diagnostics(logits, logits, chosen, top_k=2)

    assert torch.allclose(metrics["overlap_ratio"], torch.ones(2))
    assert torch.allclose(metrics["sampled_advantage"], torch.zeros(2))
    assert torch.allclose(metrics["entropy_gap"], torch.zeros(2))
    assert torch.allclose(metrics["overlap_token_advantage"], torch.zeros(2), atol=1e-7)


def test_disjoint_topk_has_zero_overlap_mass():
    student = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    teacher = torch.tensor([[0.0, 0.0, 9.0, 10.0]])
    metrics = compute_token_diagnostics(student, teacher, torch.tensor([0]), top_k=2)

    assert metrics["overlap_ratio"].item() == 0
    assert metrics["student_overlap_mass"].item() == 0
    assert metrics["teacher_overlap_mass"].item() == 0


def test_position_summary_uses_one_indexed_inclusive_bins():
    metrics = {"sampled_advantage": torch.arange(1, 6, dtype=torch.float32)}
    summary = summarize_token_diagnostics(metrics, position_bins=((1, 2), (3, 5)))

    assert summary["1-2"]["sampled_advantage"] == 1.5
    assert summary["3-5"]["sampled_advantage"] == 4.0
    assert summary["overall"]["tokens"] == 5
