import importlib.util
import json
from argparse import Namespace
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "run_opd_post_training.py"
SPEC = importlib.util.spec_from_file_location("run_opd_post_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_report_builds_tables_and_plots(tmp_path):
    for step in (0, 1):
        _append(
            tmp_path / "outcomes.jsonl",
            {
                "step": step,
                "suite": "diagnostic_greedy",
                "dataset": "math500_diagnostic",
                "problem_id": "p1",
                "generation_index": 0,
                "correct": step == 1,
                "valid_answer": True,
                "response_tokens": 10 + step,
                "truncated": False,
                "training_overlap_semantic": False,
            },
        )
        metric_values = {
            "overlap_ratio": 0.5 + 0.1 * step,
            "student_overlap_mass": 0.8,
            "teacher_overlap_mass": 0.8,
            "overlap_token_advantage": -0.01,
            "student_entropy": 1.0,
            "teacher_entropy": 1.1,
            "entropy_gap": 0.1,
            "sampled_advantage": 0.2,
            "sampled_advantage_std": 0.3,
            "sampled_advantage_positive": 0.6,
            "sampled_advantage_clip_fraction": 0.0,
            "tokens": 10,
        }
        _append(
            tmp_path / "distributional_diagnostics.jsonl",
            {
                "step": step,
                "source": "math500",
                "problem_id": "p1",
                "metrics": {
                    "overall": metric_values,
                    "1-256": metric_values,
                    "257-512": metric_values,
                    "513-1024": metric_values,
                    "1025-2048": metric_values,
                    "2049-4096": metric_values,
                },
            },
        )

    MODULE.build_report(Namespace(output_dir=tmp_path, seed=42))
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "outcome_summary.csv").exists()
    assert (tmp_path / "checkpoint_accuracy.png").exists()
    assert (tmp_path / "overlap_ratio_heatmap.png").exists()
