# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U

# Experimental Qwen3.5 2B <- 9B Lightning-OPD config.
#
# This uses the HF/FSDP backend instead of the repo's Megatron Qwen3 configs.
# It is intended for a 1-GPU pilot first; increase batch/token settings only
# after the full data path is verified.
#
# Required env vars:
#   SFT_CHECKPOINT     - path to the SFT checkpoint in HF/LlamaFactory format
#   LIGHTNING_OPD_DATA - parquet from prepare_lightning_opd.py with teacher_log_probs
#
# Optional env vars:
#   NUM_GPUS           - actor GPUs on this node (default: 1)
#   SAVE_DIR           - FSDP checkpoint directory
#   NUM_ROLLOUT        - training rollout steps (default: 200)
#   ROLLOUT_BATCH_SIZE - prompts per rollout step (default: 8)
#   GLOBAL_BATCH_SIZE  - train global batch size (default: 8)
#   MAX_TOKENS_PER_GPU - dynamic batch token cap (default: 4096)
#   USE_TENSORBOARD    - set to 1 to write TensorBoard metrics
#   TENSORBOARD_DIR    - TensorBoard output directory
#   DUMP_DETAILS       - optional debug dump directory

NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
SFT_CHECKPOINT = os.environ["SFT_CHECKPOINT"]
LIGHTNING_OPD_DATA = os.environ["LIGHTNING_OPD_DATA"]


def _get_int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _get_str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def execute(rerun=True):
    save_dir = os.environ.get("SAVE_DIR", "checkpoints/qwen35_2b_9b/lightning_opd_fsdp")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ckpt_args = (
        f"--hf-checkpoint {SFT_CHECKPOINT} "
        f"--ref-load {SFT_CHECKPOINT} "
        f"--load {save_dir} "
        f"--save {save_dir} "
        f"--save-interval {_get_int_env('SAVE_INTERVAL', 10)} "
    )

    rollout_args = (
        f"--prompt-data {LIGHTNING_OPD_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {_get_int_env('NUM_ROLLOUT', 200)} "
        f"--rollout-batch-size {_get_int_env('ROLLOUT_BATCH_SIZE', 8)} "
        "--n-samples-per-prompt 1 "
        f"--rollout-max-response-len {_get_int_env('ROLLOUT_MAX_RESPONSE_LEN', 2048)} "
        f"--global-batch-size {_get_int_env('GLOBAL_BATCH_SIZE', 8)} "
    )

    rm_args = (
        "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
        "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
        "--include-verifiable-reward "
    )

    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        f"--attn-implementation {_get_str_env('FSDP_ATTN_IMPLEMENTATION', 'sdpa')} "
        "--optimizer adam "
        "--lr 2e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {_get_int_env('MAX_TOKENS_PER_GPU', 4096)} "
    )

    grpo_args = (
        "--advantage-estimator on_policy_distillation "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
    )

    wandb_args = ""
    if os.environ.get("WANDB_KEY"):
        wandb_args = (
            "--use-wandb "
            "--wandb-project lightning-opd "
            "--wandb-group qwen35-2b-lightning-opd-fsdp "
            f"--wandb-key {os.environ['WANDB_KEY']} "
        )

    tensorboard_args = ""
    if os.environ.get("USE_TENSORBOARD", "0") == "1":
        tensorboard_args = "--use-tensorboard "
        if os.environ.get("TENSORBOARD_DIR"):
            tensorboard_args += f"--tensorboard-dir {os.environ['TENSORBOARD_DIR']} "

    debug_args = ""
    if os.environ.get("DUMP_DETAILS"):
        debug_args = f"--dump-details {os.environ['DUMP_DETAILS']} "

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--rollout-num-gpus 0 "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {NUM_GPUS} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{rm_args} "
        f"{grpo_args} "
        f"{fsdp_args} "
        f"{wandb_args} "
        f"{tensorboard_args} "
        f"{debug_args} "
        f"{misc_args} "
    )

    U.execute_train(
        rerun=rerun,
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
    )


if __name__ == "__main__":
    execute(rerun=False)
