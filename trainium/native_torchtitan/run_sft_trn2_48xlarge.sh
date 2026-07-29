#!/usr/bin/env bash
# Launch native Qwen3-4B SFT on one trn2.48xlarge instance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${SFT_DATA:?Set SFT_DATA to the parquet file/directory on this instance}"
: "${MODEL_DIR:?Set MODEL_DIR to the local Qwen/Qwen3-4B-Base snapshot}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to checkpoint storage on this instance}"
: "${RUN_ID:?Set a stable RUN_ID for this run}"

TORCHTITAN_DIR="${TORCHTITAN_DIR:-${HOME}/torchtitan}"
TORCHTITAN_COMMIT="0a2107f984639e23a0e5b07fc278785345f03b73"
CONFIG_FILE="${SCRIPT_DIR}/qwen3_4b_sft_tp4_fsdp_trn2_48xlarge.toml"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-${RUN_ID}}"
RUN_DIR="${OUTPUT_ROOT%/}/${RUN_ID}"
SMOKE="${SMOKE:-0}"
EXPECTED_SFT_ROWS="${EXPECTED_SFT_ROWS:-300000}"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NEURON_RT_NUM_CORES=64
export TORCH_NEURONX_ENABLE_HOST_CC="${TORCH_NEURONX_ENABLE_HOST_CC:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-4b-base-open-thoughts3-qwen3-8b}"
unset NEURON_RT_VISIBLE_CORES
unset NEURON_VISIBLE_DEVICES

if [[ "${TORCH_NEURONX_ENABLE_HOST_CC}" != "0" && \
      "${TORCH_NEURONX_ENABLE_HOST_CC}" != "1" ]]; then
    echo "ERROR: TORCH_NEURONX_ENABLE_HOST_CC must be 0 or 1." >&2
    exit 1
fi

for required_path in "${SFT_DATA}" "${MODEL_DIR}" "${OUTPUT_ROOT}" "${CONFIG_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
for model_asset in config.json tokenizer.json tokenizer_config.json; do
    if [[ ! -f "${MODEL_DIR}/${model_asset}" ]]; then
        echo "ERROR: MODEL_DIR is missing ${model_asset}: ${MODEL_DIR}" >&2
        exit 1
    fi
done
if ! compgen -G "${MODEL_DIR}/*.safetensors" >/dev/null; then
    echo "ERROR: MODEL_DIR has no safetensors weights: ${MODEL_DIR}" >&2
    exit 1
fi

python3 - "${SFT_DATA}" "${EXPECTED_SFT_ROWS}" "${MODEL_DIR}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoConfig

dataset_path = Path(sys.argv[1]).expanduser()
expected_rows = int(sys.argv[2])
model_path = sys.argv[3]
parquet_files = (
    [dataset_path]
    if dataset_path.is_file()
    else sorted(dataset_path.rglob("*.parquet"))
)
if not parquet_files:
    raise SystemExit(f"No parquet files found under {dataset_path}")

rows = 0
for parquet_file in parquet_files:
    parquet = pq.ParquetFile(parquet_file)
    if "messages" not in parquet.schema_arrow.names:
        raise SystemExit(f"{parquet_file} has no messages column")
    rows += parquet.metadata.num_rows
if rows != expected_rows:
    raise SystemExit(
        f"Found {rows} SFT rows across {len(parquet_files)} parquet files; "
        f"expected {expected_rows}. Check SFT_DATA against the generated dataset."
    )

config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=False,
    local_files_only=True,
)
if config.model_type != "qwen3":
    raise SystemExit(f"Expected a qwen3 checkpoint, found {config.model_type!r}")
if config.hidden_size != 2560 or config.num_hidden_layers != 36:
    raise SystemExit(
        "MODEL_DIR is not Qwen3-4B-Base: "
        f"hidden_size={config.hidden_size}, layers={config.num_hidden_layers}"
    )
if config.eos_token_id != 151643 or config.max_position_embeddings != 32768:
    raise SystemExit(
        "MODEL_DIR appears to be Qwen3-4B rather than Qwen3-4B-Base: "
        f"eos={config.eos_token_id}, max_positions={config.max_position_embeddings}"
    )
print(f"Validated {rows} SFT rows in {len(parquet_files)} parquet file(s).")
print("Validated Qwen3-4B-Base model configuration.")
PY

if [[ ! -d "${TORCHTITAN_DIR}/.git" ]]; then
    echo "ERROR: TorchTitan checkout not found at ${TORCHTITAN_DIR}." >&2
    echo "Run trainium/native_torchtitan/setup_torchtitan.sh first." >&2
    exit 1
fi
if [[ "$(git -C "${TORCHTITAN_DIR}" rev-parse HEAD)" != "${TORCHTITAN_COMMIT}" ]]; then
    echo "ERROR: TorchTitan is not at the pinned native-Qwen3 commit." >&2
    exit 1
fi
PATCH_FILE="${TORCH_NEURONX_SRC:-${HOME}/torch-neuronx}/docs/torchtitan/qwen3/TorchTitan.diff"
if [[ ! -f "${PATCH_FILE}" ]] || \
   ! git -C "${TORCHTITAN_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "ERROR: the torch-neuronx Qwen3 patch has not been applied to TorchTitan." >&2
    echo "Run trainium/native_torchtitan/setup_torchtitan.sh first." >&2
    exit 1
fi

if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "ERROR: neuron-ls is unavailable; use the native TorchNeuron beta environment." >&2
    exit 1
fi
neuron-ls

python3 - <<'PY'
import torch
import torch_neuronx

assert torch.device("neuron").type == "neuron"
assert torch_neuronx.device_count() == 64, (
    f"Expected 64 logical Neuron devices, found {torch_neuronx.device_count()}"
)
assert "neuron" in getattr(torch.distributed.Backend, "backend_list", ())
print(f"Torch {torch.__version__}; torch-neuronx {torch_neuronx.__version__}; 64 devices")
PY

# A new run loads the HF base checkpoint. A populated run resumes its newest
# DCP checkpoint. An empty checkpoint directory is ambiguous in TorchTitan and
# could otherwise skip the HF initialization.
if [[ -d "${RUN_DIR}/checkpoint" ]] && \
   ! find "${RUN_DIR}/checkpoint" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print -quit |
       grep -q .; then
    echo "ERROR: ${RUN_DIR}/checkpoint exists but contains no resumable step." >&2
    echo "Choose a new RUN_ID or move the empty checkpoint directory aside." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${TORCHTITAN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_OVERRIDES=(
    --job.dump_folder "${RUN_DIR}"
    --model.hf_assets_path "${MODEL_DIR}"
    --training.dataset_path "${SFT_DATA}"
)
if [[ "${SMOKE}" == "1" ]]; then
    TRAIN_OVERRIDES+=(
        --training.steps 2
        --training.global_batch_size 16
        --checkpoint.load-only
    )
elif [[ "${SMOKE}" != "0" ]]; then
    echo "ERROR: SMOKE must be 0 or 1." >&2
    exit 1
fi

echo "=== Native TorchNeuron Qwen3-4B SFT ==="
echo "Host:          $(hostname)"
echo "Rendezvous:    127.0.0.1:${MASTER_PORT} (${RDZV_ID})"
echo "Topology:      TP=4 x FSDP=16 over 64 native Neuron processes"
echo "Dataset:       ${SFT_DATA}"
echo "Base model:    ${MODEL_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "W&B project:   ${WANDB_PROJECT}"
echo "W&B run name:  ${WANDB_RUN_NAME}"
echo "Host CC:       ${TORCH_NEURONX_ENABLE_HOST_CC}"
echo "Smoke:         ${SMOKE}"
echo "========================================"

torchrun \
    --nnodes 1 \
    --nproc_per_node 64 \
    --rdzv_id "${RDZV_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    "${SCRIPT_DIR}/train_sft.py" \
    --job.config_file "${CONFIG_FILE}" \
    "${TRAIN_OVERRIDES[@]}"
