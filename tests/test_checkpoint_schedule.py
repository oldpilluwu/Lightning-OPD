from argparse import Namespace

import pytest

from slime.utils.checkpoint_schedule import parse_step_list, scheduled_checkpoint, validate_checkpoint_schedule


def test_parse_step_list_is_sorted_and_unique():
    assert parse_step_list("10,2,2,1") == (1, 2, 10)
    assert parse_step_list("") == ()


def test_parse_step_list_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        parse_step_list("0,1")


def test_scheduled_checkpoint_separates_weights_and_optimizer():
    checkpoints = (1, 2, 50, 75, 100, 150)
    optimizer = (50, 100, 150)
    assert scheduled_checkpoint(1, checkpoints, optimizer) == (True, False)
    assert scheduled_checkpoint(50, checkpoints, optimizer) == (True, True)
    assert scheduled_checkpoint(51, checkpoints, optimizer) == (False, False)


def test_validate_checkpoint_schedule_rejects_optimizer_only_step(tmp_path):
    args = Namespace(
        checkpoint_steps=(1, 50),
        optimizer_checkpoint_steps=(25, 50),
        save=str(tmp_path),
        save_interval=None,
        async_save=False,
        num_rollout=150,
    )
    with pytest.raises(ValueError, match="also be checkpoint steps"):
        validate_checkpoint_schedule(args)
