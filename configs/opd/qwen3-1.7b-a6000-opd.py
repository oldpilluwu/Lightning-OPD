# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-A6000 standard OPD: Qwen3-1.7B-SFT student and Qwen3-4B teacher.

The teacher is a live SGLang server on GPU 0. The Megatron actor and student
rollout engine share that GPU using slime's colocated offload path.
"""

import os
import random
import string
from pathlib import Path

import slime.utils.external_utils.command_utils as U
from slime.utils.external_utils.command_utils import get_bool_env_var

MODEL_NAME = os.environ.get("OPD_MODEL_NAME", "Qwen3-1.7B-SFT-Qwen3-4B-OPD")
MODEL_TYPE = "qwen3-1.7B"
NUM_GPUS = 1

STUDENT_MODEL_ID = os.environ.get("STUDENT_MODEL_ID", "lllyx/Qwen3-1.7B-SFT")
TEACHER_MODEL_ID = os.environ.get("TEACHER_MODEL_ID", "Qwen/Qwen3-4B")
STUDENT_MODEL_REVISION = os.environ.get(
    "STUDENT_MODEL_REVISION", "f0065babe7440257d7db331a5aa442ef7c3cf266"
)
TEACHER_MODEL_REVISION = os.environ.get(
    "TEACHER_MODEL_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"
)
SFT_CHECKPOINT = os.environ.get("SFT_CHECKPOINT", "/root/models/Qwen3-1.7B-SFT")
TEACHER_CHECKPOINT = os.environ.get("TEACHER_CHECKPOINT", "/root/models/Qwen3-4B")
PROMPT_DATA = os.environ.get("OPD_PROMPT_DATA", "/root/datasets/qwen3-1.7b-opd/dapo_train.jsonl")

TEACHER_IP = os.environ.get("MASTER_ADDR", "127.0.0.1")
TEACHER_PORT = int(os.environ.get("OPD_TEACHER_PORT", "13141"))
TEACHER_MEM_FRACTION = float(os.environ.get("OPD_TEACHER_MEM_FRACTION", "0.22"))
ROLLOUT_MEM_FRACTION = float(os.environ.get("OPD_ROLLOUT_MEM_FRACTION", "0.14"))
SERVER_CONCURRENCY = int(os.environ.get("OPD_SERVER_CONCURRENCY", "2"))
MAX_RESPONSE_LEN = int(os.environ.get("OPD_MAX_RESPONSE_LEN", "4096"))
MAX_PROMPT_LEN = int(os.environ.get("OPD_MAX_PROMPT_LEN", "1024"))
MAX_CONTEXT_LEN = int(os.environ.get("OPD_MAX_CONTEXT_LEN", str(MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1024)))
SMOKE_TEST = get_bool_env_var("OPD_SMOKE_TEST", "0")

CHECKPOINT_STEPS = "1,2,3,4,5,10,15,20,25,30,50,75,100,125,150"
OPTIMIZER_CHECKPOINT_STEPS = "50,100,150"


def _download_models_and_data() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets/qwen3-1.7b-opd")
    if not Path(SFT_CHECKPOINT).exists():
        U.exec_command(
            f"hf download {STUDENT_MODEL_ID} --revision {STUDENT_MODEL_REVISION} --local-dir {SFT_CHECKPOINT}"
        )
    if not Path(TEACHER_CHECKPOINT).exists():
        U.exec_command(
            f"hf download {TEACHER_MODEL_ID} --revision {TEACHER_MODEL_REVISION} --local-dir {TEACHER_CHECKPOINT}"
        )
    if not Path(PROMPT_DATA).exists():
        U.exec_command(
            "python scripts/prepare_qwen3_1p7b_opd_data.py "
            f"--output-dir {Path(PROMPT_DATA).parent}"
        )


def deploy_teacher_model() -> None:
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    log_file = f"/tmp/sglang_qwen3_4b_teacher_{suffix}.log"
    external_ray = get_bool_env_var("SLIME_SCRIPT_EXTERNAL_RAY")

    U.exec_command(
        "pkill -9 -f 'sglang.launch_server.*Qwen3-4B' || true; "
        f"{'' if external_ray else 'ray stop --force; '}"
        "sleep 3"
    )
    U.exec_command(
        "CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server "
        f"--model-path {TEACHER_CHECKPOINT} "
        "--host 0.0.0.0 "
        f"--port {TEACHER_PORT} "
        "--tp 1 "
        f"--chunked-prefill-size {MAX_RESPONSE_LEN} "
        f"--mem-fraction-static {TEACHER_MEM_FRACTION} "
        f"--context-length {MAX_CONTEXT_LEN} "
        f"--max-running-requests {SERVER_CONCURRENCY} "
        "--disable-cuda-graph "
        f"> {log_file} 2>&1 &"
    )
    U.exec_command(
        f"until curl -sf http://{TEACHER_IP}:{TEACHER_PORT}/health_generate >/dev/null; do "
        f"echo 'Waiting for Qwen3-4B teacher...'; tail -n 20 {log_file}; sleep 5; done"
    )


def prepare() -> None:
    _download_models_and_data()
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        hf_checkpoint=SFT_CHECKPOINT,
    )
    deploy_teacher_model()


def execute(rerun: bool = False) -> None:
    save_path = f"/root/models/{MODEL_NAME}_ckpt__{Path(__file__).stem}/"
    num_rollout = 2 if SMOKE_TEST else 150
    rollout_batch_size = 2 if SMOKE_TEST else 16
    response_len = 512 if SMOKE_TEST else MAX_RESPONSE_LEN
    max_tokens_per_gpu = max(MAX_PROMPT_LEN + response_len, 2048)

    ckpt_args = (
        f"--hf-checkpoint {SFT_CHECKPOINT} "
        f"--load /root/models/{MODEL_NAME}_torch_dist "
        f"--save {save_path} "
    )
    if not SMOKE_TEST:
        ckpt_args += (
            f"--checkpoint-steps {CHECKPOINT_STEPS} "
            f"--optimizer-checkpoint-steps {OPTIMIZER_CHECKPOINT_STEPS} "
        )

    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        "--rollout-shuffle "
        "--rollout-seed 42 "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        "--n-samples-per-prompt 1 "
        f"--rollout-max-prompt-len {MAX_PROMPT_LEN} "
        f"--rollout-max-response-len {response_len} "
        f"--rollout-max-context-len {MAX_CONTEXT_LEN} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k -1 "
        f"--global-batch-size {rollout_batch_size} "
    )
    rm_args = (
        "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
        "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
        f"--rm-url http://{TEACHER_IP}:{TEACHER_PORT}/generate "
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
    algo_args = (
        "--advantage-estimator on_policy_distillation "
        "--opd-advantage-clip 10 "
        f"--opd-student-model-id {STUDENT_MODEL_ID} "
        f"--opd-student-model-revision {STUDENT_MODEL_REVISION} "
        f"--opd-teacher-model-id {TEACHER_MODEL_ID} "
        f"--opd-teacher-model-revision {TEACHER_MODEL_REVISION} "
        "--kl-coef 0 "
        "--entropy-coef 0 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--seed 42 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {ROLLOUT_MEM_FRACTION} "
        f"--sglang-context-length {MAX_CONTEXT_LEN} "
        f"--sglang-max-running-requests {SERVER_CONCURRENCY} "
        f"--sglang-server-concurrency {SERVER_CONCURRENCY} "
        "--sglang-disable-cuda-graph "
    )
    misc_args = (
        "--attention-dropout 0 "
        "--hidden-dropout 0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 "
        "--num-gpus-per-node 1 "
        "--colocate "
    )
    wandb_args = ""
    if os.environ.get("WANDB_KEY"):
        wandb_args = (
            "--use-wandb --wandb-project lightning-opd "
            f"--wandb-group {Path(__file__).stem} --wandb-key {os.environ['WANDB_KEY']} "
        )

    U.execute_train(
        rerun=rerun,
        train_args=(
            ckpt_args
            + rollout_args
            + rm_args
            + perf_args
            + algo_args
            + optimizer_args
            + sglang_args
            + misc_args
            + wandb_args
        ),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    execute(rerun=False)
