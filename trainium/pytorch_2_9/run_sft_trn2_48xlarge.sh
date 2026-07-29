#!/usr/bin/env bash
# Direct PyTorch 2.9/XLA/NxD full SFT on one trn2.48xlarge.

set -euo pipefail

: "${TOKENIZED_DATA:?Set TOKENIZED_DATA to prepare_sft_dataset.py output}"
: "${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to prepare_model_checkpoint.sh output}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16384}"
SMOKE="${SMOKE:-0}"
RESUME_TAG="${RESUME_TAG:-}"

if [[ -z "${VIRTUAL_ENV:-}" || \
      "$(basename -- "${VIRTUAL_ENV}")" != "aws_neuronx_venv_pytorch_2_9" ]]; then
    echo "ERROR: activate aws_neuronx_venv_pytorch_2_9 first." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${SCRIPT_DIR}"
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS_OVERRIDE:---model-type transformer --distribution-strategy llm-training --logical-nc-config 2 --retry_failed_compilation}"
export NEURON_RT_STOCHASTIC_ROUNDING_EN=0
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS="${NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS:-3}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-64}"
export TOKENIZERS_PARALLELISM=false

if [[ "${SMOKE}" == "1" ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
    MAX_STEPS="${MAX_STEPS:-2}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
    REPORT_TO="${REPORT_TO:-none}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-64}"
    MAX_STEPS="${MAX_STEPS:-3000}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
    SAVE_STEPS="${SAVE_STEPS:-100}"
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
    REPORT_TO="${REPORT_TO:-wandb}"
fi

TP_DEGREE="${TP_DEGREE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
if (( NPROC_PER_NODE % TP_DEGREE != 0 )); then
    echo "ERROR: process count must be divisible by TP degree." >&2
    exit 1
fi
if (( SEQUENCE_LENGTH % 2048 != 0 )); then
    echo "ERROR: sequence length must be a multiple of 2048 for NKI flash attention." >&2
    exit 1
fi

DP_DEGREE=$((NPROC_PER_NODE / TP_DEGREE))
GLOBAL_BATCH=$((DP_DEGREE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${SMOKE}" != "1" && "${GLOBAL_BATCH}" -ne 256 ]]; then
    echo "ERROR: production global batch must be 256; got ${GLOBAL_BATCH}." >&2
    exit 1
fi

PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
TORCHRUN_BIN="${VIRTUAL_ENV}/bin/torchrun"
"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
from packaging.version import Version
for package in ("torch", "torch-xla", "torch-neuronx"):
    if Version(version(package)).release[:2] != (2, 9):
        raise SystemExit(f"{package} {version(package)} is not PyTorch 2.9")
nxd = Version(version("neuronx-distributed"))
if not (Version("0.19") <= nxd < Version("0.20")):
    raise SystemExit(f"neuronx-distributed {nxd} is not the required 0.19 line")
PY

mkdir -p "${OUTPUT_DIR}"
EXTRA_ARGS=()
if [[ -n "${RESUME_TAG}" ]]; then
    EXTRA_ARGS+=(--resume-tag "${RESUME_TAG}")
fi
if [[ "${SMOKE}" == "1" ]]; then
    EXTRA_ARGS+=(--smoke)
fi

echo "=== Direct PyTorch 2.9 Qwen3 full SFT ==="
echo "Processes:    ${NPROC_PER_NODE}"
echo "TP / DP:      ${TP_DEGREE} / ${DP_DEGREE}"
echo "Sequence:     ${SEQUENCE_LENGTH}"
echo "Global batch: ${GLOBAL_BATCH}"
echo "Mode:         FULL MODEL (no PEFT/LoRA/Optimum)"

exec "${TORCHRUN_BIN}" \
    --nnodes 1 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:0" \
    "${SCRIPT_DIR}/train_sft.py" \
    --model-id "${MODEL_ID}" \
    --tokenized-dataset "${TOKENIZED_DATA}" \
    --pretrained-checkpoint "${MODEL_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --tensor-parallel-size "${TP_DEGREE}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate 8e-5 \
    --warmup-ratio 0.1 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-epsilon 1e-8 \
    --max-grad-norm 1.0 \
    --save-steps "${SAVE_STEPS}" \
    --save-total-limit 10 \
    --logging-steps 1 \
    --seed 42 \
    --data-seed 42 \
    --dataloader-workers "${DATALOADER_WORKERS}" \
    --ddp-timeout 180000000 \
    --report-to "${REPORT_TO}" \
    "${EXTRA_ARGS[@]}"
