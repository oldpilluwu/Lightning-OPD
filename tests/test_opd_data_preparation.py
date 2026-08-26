import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_qwen3_1p7b_opd_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_opd_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_dapo_preserves_messages_and_ground_truth():
    row = {
        "prompt": [{"role": "user", "content": "problem"}],
        "reward_model": {"ground_truth": "42"},
        "extra_info": {"index": "abc"},
    }
    normalized = MODULE._normalize_dapo(row)
    assert normalized == {
        "id": "abc",
        "prompt": [{"role": "user", "content": "problem"}],
        "label": "42",
        "source": "dapo",
    }


def test_math500_stratification_is_deterministic():
    rows = [
        {"id": str(index), "subject": f"subject-{index % 3}", "level": index % 2}
        for index in range(30)
    ]
    first = MODULE.stratified_math500(rows, 12, 42)
    second = MODULE.stratified_math500(rows, 12, 42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["subject"] for row in first}) == 3
