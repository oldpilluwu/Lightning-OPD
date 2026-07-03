# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U

# Single-GPU Lightning OPD for Qwen3-4B student / Qwen3-8B teacher.
#
# Required env vars:
#   SFT_CHECKPOINT     - HF-format SFT checkpoint
#   LIGHTNING_OPD_DATA - precomputed Lightning OPD parquet
#
# Optional env vars:
#   LIGHTNING_OPD_SAVE_DIR
#   LIGHTNING_OPD_NUM_ROLLOUT
#   LIGHTNING_OPD_BATCH_SIZE
#   LIGHTNING_OPD_MAX_TOKENS_PER_GPU
#   LIGHTNING_OPD_SAVE_INTERVAL
#   LIGHTNING_OPD_LR

MODEL_NAME = "Qwen3-4B-Base-Open-Thoughts-Qwen3-8B-sft-single-gpu"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 1
SFT_CHECKPOINT = os.environ["SFT_CHECKPOINT"]


def prepare():
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        hf_checkpoint=SFT_CHECKPOINT,
    )


def execute(rerun=True):
    load_save_path = os.environ.get(
        "LIGHTNING_OPD_SAVE_DIR",
        f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}/",
    )
    batch_size = int(os.environ.get("LIGHTNING_OPD_BATCH_SIZE", "8"))
    num_rollout = int(os.environ.get("LIGHTNING_OPD_NUM_ROLLOUT", "250"))
    save_interval = int(os.environ.get("LIGHTNING_OPD_SAVE_INTERVAL", "25"))
    max_response_len = int(os.environ.get("LIGHTNING_OPD_MAX_RESPONSE_LEN", "4096"))
    max_tokens_per_gpu = int(os.environ.get("LIGHTNING_OPD_MAX_TOKENS_PER_GPU", "8192"))
    lr = os.environ.get("LIGHTNING_OPD_LR", "2e-6")

    ckpt_args = (
        f"--hf-checkpoint {SFT_CHECKPOINT} "
        f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {save_interval} "
        "--save-retain-interval 100 "
    )

    rollout_args = (
        f"--prompt-data {os.environ['LIGHTNING_OPD_DATA']} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {batch_size} "
        "--n-samples-per-prompt 1 "
        f"--rollout-max-response-len {max_response_len} "
        f"--global-batch-size {batch_size} "
    )

    rm_args = (
        "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
        "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
        "--include-verifiable-reward "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator on_policy_distillation "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    wandb_args = ""
    if os.environ.get("WANDB_KEY"):
        wandb_args = (
            "--use-wandb "
            "--wandb-project lightning-opd "
            f"--wandb-group {Path(__file__).stem} "
            f"--wandb-key {os.environ['WANDB_KEY']} "
        )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.4 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--num-gpus-per-node 1 "
        "--rollout-num-gpus 0 "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{rm_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{wandb_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        rerun=rerun,
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute(rerun=False)
