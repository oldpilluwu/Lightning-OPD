#!/usr/bin/env bash
# Full-parameter Qwen3-4B-Base SFT on one trn2.48xlarge.

set -euo pipefail

: "${TOKENIZED_DATA:?Set TOKENIZED_DATA to prepare_sft_dataset.py output}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
OPTIMUM_NEURON_SRC="${OPTIMUM_NEURON_SRC:-${HOME}/optimum-neuron-qwen3-pt29}"
OPTIMUM_NEURON_REF="4a80f2f3de15e83e978a6f3c0d43224626d921ca"
export OPTIMUM_NEURON_SRC
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16384}"
SMOKE="${SMOKE:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ ! -d "${OPTIMUM_NEURON_SRC}/optimum/neuron" ]]; then
    echo "ERROR: audited Optimum Neuron source not found: ${OPTIMUM_NEURON_SRC}" >&2
    echo "Run trainium/optimum_neuron/setup_instance.sh first." >&2
    exit 1
fi
if [[ ! -d "${OPTIMUM_NEURON_SRC}/.git" ]] || \
   [[ "$(git -C "${OPTIMUM_NEURON_SRC}" rev-parse HEAD)" != "${OPTIMUM_NEURON_REF}" ]]; then
    echo "ERROR: Optimum Neuron source is not at audited commit ${OPTIMUM_NEURON_REF}." >&2
    echo "Run trainium/optimum_neuron/setup_instance.sh first." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${OPTIMUM_NEURON_SRC}:${SCRIPT_DIR}"
export PJRT_DEVICE="${PJRT_DEVICE:-NEURON}"
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type transformer --retry_failed_compilation -lnc 2}"
export NEURON_FUSE_SOFTMAX="${NEURON_FUSE_SOFTMAX:-1}"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS="${NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS:-3}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-64}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${SMOKE}" == "1" ]]; then
    # Trn2 with its default LNC=2 supports 1, 4, 16, or 64 XLA workers.
    # TP8 therefore needs at least 16 workers (two data-parallel groups).
    NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
    TP_DEGREE="${TP_DEGREE:-8}"
    MAX_STEPS="${MAX_STEPS:-2}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    REPORT_TO="${REPORT_TO:-none}"
elif [[ "${NEURON_EXTRACT_GRAPHS_ONLY:-0}" == "1" ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-64}"
    TP_DEGREE="${TP_DEGREE:-8}"
    MAX_STEPS="${MAX_STEPS:-2}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    REPORT_TO="${REPORT_TO:-none}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-64}"
    TP_DEGREE="${TP_DEGREE:-8}"
    MAX_STEPS="${MAX_STEPS:-3000}"
    # Original global batch: 32 GPU DP ranks * 4 microbatch * 2 accumulation = 256.
    # Trainium: 64 processes / TP8 = 8 DP ranks; 8 * 1 * 32 = the same 256.
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
    SAVE_STEPS="${SAVE_STEPS:-100}"
    REPORT_TO="${REPORT_TO:-wandb}"
fi

if [[ "${NEURON_LOGICAL_NC_CONFIG}" != "2" ]]; then
    echo "ERROR: this trn2.48xlarge launcher requires NEURON_LOGICAL_NC_CONFIG=2." >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" || \
      "$(basename -- "${VIRTUAL_ENV}")" != "aws_neuronx_venv_pytorch_2_9" ]]; then
    echo "ERROR: activate aws_neuronx_venv_pytorch_2_9 before launching." >&2
    exit 1
fi
PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
TORCHRUN_BIN="${VIRTUAL_ENV}/bin/torchrun"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "ERROR: the active PyTorch 2.9 Neuron venv is incomplete: ${VIRTUAL_ENV}" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
import os
import pathlib

required_versions = {
    "torch": (2, 9),
    "torch-xla": (2, 9),
    "torch-neuronx": (2, 9),
}
for package, expected_version in required_versions.items():
    try:
        actual = version(package)
    except PackageNotFoundError as error:
        raise SystemExit(f"{package} is missing; activate aws_neuronx_venv_pytorch_2_9") from error
    if Version(actual).release[:2] != expected_version:
        raise SystemExit(
            f"{package} {actual} is active; training requires aws_neuronx_venv_pytorch_2_9"
        )
nxd_version = Version(version("neuronx-distributed"))
if not (Version("0.18") <= nxd_version < Version("0.20")):
    raise SystemExit(
        f"neuronx-distributed {nxd_version} is outside the supported "
        "PyTorch 2.9 DLAMI range (0.18.x or 0.19.x)"
    )

import optimum.neuron
import neuronx_distributed
import torch_xla
from optimum.neuron import NeuronTrainingArguments

source = pathlib.Path(optimum.neuron.__file__).resolve()
expected = pathlib.Path(os.environ["OPTIMUM_NEURON_SRC"]).expanduser().resolve()
if expected not in source.parents:
    raise SystemExit(f"Unexpected optimum.neuron source: {source}")
required_arguments = {
    "tensor_parallel_size",
    "disable_sequence_parallel",
    "zero_1",
    "stochastic_rounding_enabled",
    "async_save",
}
missing = required_arguments - set(NeuronTrainingArguments.__dataclass_fields__)
if missing:
    raise SystemExit(f"Incompatible NeuronTrainingArguments API: {sorted(missing)}")
print("PyTorch 2.9 Neuron/XLA preflight passed.")
print("Optimum Neuron source:", source)
PY

if (( NPROC_PER_NODE % TP_DEGREE != 0 )); then
    echo "ERROR: NPROC_PER_NODE must be divisible by TP_DEGREE." >&2
    exit 1
fi

DP_DEGREE=$((NPROC_PER_NODE / TP_DEGREE))
GLOBAL_BATCH=$((DP_DEGREE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${SMOKE}" != "1" && "${GLOBAL_BATCH}" -ne 256 ]]; then
    echo "ERROR: Production global batch must remain 256; got ${GLOBAL_BATCH}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "=== Optimum Neuron Qwen3 full SFT ==="
echo "Model:          ${MODEL_ID}"
echo "Dataset:        ${TOKENIZED_DATA}"
echo "Output:         ${OUTPUT_DIR}"
echo "Processes:      ${NPROC_PER_NODE}"
echo "TP / DP:        ${TP_DEGREE} / ${DP_DEGREE}"
echo "Sequence:       ${SEQUENCE_LENGTH}"
echo "Steps:          ${MAX_STEPS}"
echo "Microbatch:     ${PER_DEVICE_BATCH_SIZE}"
echo "Grad accum:     ${GRADIENT_ACCUMULATION_STEPS}"
echo "Global batch:   ${GLOBAL_BATCH}"
echo "Trainable mode: FULL MODEL (no PEFT/LoRA)"

exec "${TORCHRUN_BIN}" \
    --nnodes 1 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:0" \
    "${SCRIPT_DIR}/sft_qwen3.py" \
    --model_id "${MODEL_ID}" \
    --tokenized_dataset "${TOKENIZED_DATA}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --do_train true \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --gradient_checkpointing false \
    --learning_rate 8e-5 \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --stochastic_rounding_enabled false \
    --tensor_parallel_size "${TP_DEGREE}" \
    --disable_sequence_parallel false \
    --zero_1 false \
    --async_save false \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 10 \
    --save_only_model false \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_pin_memory true \
    --ddp_timeout 180000000 \
    --seed 42 \
    --data_seed 42 \
    --report_to "${REPORT_TO}" \
    --run_name qwen3-4b-base-open-thoughts3-qwen3-8b \
    --overwrite_output_dir false \
    "${EXTRA_ARGS[@]}"
