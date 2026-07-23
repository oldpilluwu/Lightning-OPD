#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Paper-faithful SFT data curation for the Lightning OPD 4B student path:
#   source:  open-thoughts/OpenThoughts3-1.2M (300K sampled prompts)
#   teacher: Qwen/Qwen3-8B
#   sampling: temperature=0.7, top_p=0.9, one response, 16K token cap
#
# Target: one or more trn2.48xlarge instances. With Trn2 LNC=2, each instance
# exposes 64 logical NeuronCores. This launcher creates 16 independent
# data-parallel replicas per instance; each replica uses four logical
# NeuronCores (TP=4), one Trainium2 chip.
#
# This is an end-to-end, resumable entrypoint: environment verification,
# dependencies, dataset preparation, local model download, compilation,
# generation, merge, and final validation all happen in one invocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 5 )); then
    echo "ERROR: Bash 5 or newer is required (the Ubuntu 24.04 DLAMI provides it)." >&2
    exit 1
fi

# Storage can be redirected to instance NVMe, which is recommended for the
# compile cache and large Arrow shards:
#   WORK_ROOT=/path/on/local/nvme bash trainium/nxd/run_sft_curation_trn2_48xlarge.sh
WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/data/trainium/nxd}"
export WORK_ROOT
export HF_HOME="${HF_HOME:-${WORK_ROOT}/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORK_ROOT}/cache/vllm}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-${WORK_ROOT}/cache/neuron-compile-cache}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.sh"

EXPECTED_INSTANCE_TYPE="trn2.48xlarge"
ALLOW_INSTANCE_MISMATCH="${ALLOW_INSTANCE_MISMATCH:-0}"

detect_instance_type() {
    local token=""
    token="$(curl -fsS --max-time 2 -X PUT \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
        http://169.254.169.254/latest/api/token 2>/dev/null || true)"
    if [[ -n "${token}" ]]; then
        curl -fsS --max-time 2 \
            -H "X-aws-ec2-metadata-token: ${token}" \
            http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true
    fi
}

INSTANCE_TYPE="$(detect_instance_type)"
if [[ -n "${INSTANCE_TYPE}" && "${INSTANCE_TYPE}" != "${EXPECTED_INSTANCE_TYPE}" ]]; then
    if [[ "${ALLOW_INSTANCE_MISMATCH}" != "1" ]]; then
        echo "ERROR: expected ${EXPECTED_INSTANCE_TYPE}, detected ${INSTANCE_TYPE}." >&2
        echo "Set ALLOW_INSTANCE_MISMATCH=1 only for deliberate topology testing." >&2
        exit 1
    fi
    echo "WARNING: expected ${EXPECTED_INSTANCE_TYPE}, detected ${INSTANCE_TYPE}." >&2
elif [[ -z "${INSTANCE_TYPE}" ]]; then
    echo "WARNING: EC2 instance metadata is unavailable; relying on neuron-ls." >&2
fi

# Semantic settings are fixed to the repository's Lightning OPD 4B paper
# recipe. Hardware scheduling knobs remain overridable for measured tuning.
TEACHER_MODEL="Qwen/Qwen3-8B"
HF_DATASET="open-thoughts/OpenThoughts3-1.2M"
SFT_SAMPLES=300000
SEED=42
MAX_TOKENS=16384
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
TEMPERATURE="0.7"
TOP_P="0.9"
NUM_RESPONSES=1
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TP_SIZE="${TP_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_INSTANCES="${NUM_INSTANCES:-1}"
INSTANCE_RANK="${INSTANCE_RANK:-0}"
PRE_SHARDED_INPUT="${PRE_SHARDED_INPUT:-0}"
PRECOMPILE="${PRECOMPILE:-1}"

EXECUTION_TAG="tp${TP_SIZE}-batch${BATCH_SIZE}-seqs${MAX_NUM_SEQS}-len${MAX_MODEL_LEN}"
export NEURON_COMPILED_ARTIFACTS="${NEURON_COMPILED_ARTIFACTS:-${WORK_ROOT}/compiled/qwen3-8b-${EXECUTION_TAG}}"

TOTAL_LOGICAL_CORES=64
for positive_integer in \
    "${TP_SIZE}" "${NUM_WORKERS}" "${NUM_INSTANCES}" \
    "${MAX_NUM_SEQS}" "${BATCH_SIZE}"; do
    if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: topology and batch settings must be positive integers." >&2
        exit 1
    fi
done
if [[ ! "${INSTANCE_RANK}" =~ ^[0-9]+$ ]] || (( INSTANCE_RANK >= NUM_INSTANCES )); then
    echo "ERROR: INSTANCE_RANK=${INSTANCE_RANK} must be in [0, $(( NUM_INSTANCES - 1 ))]." >&2
    exit 1
fi
if (( NUM_WORKERS * TP_SIZE > TOTAL_LOGICAL_CORES )); then
    echo "ERROR: NUM_WORKERS=${NUM_WORKERS} and TP_SIZE=${TP_SIZE} exceed 64 LNC2 cores." >&2
    exit 1
fi
if (( MAX_MODEL_LEN <= MAX_TOKENS )); then
    echo "ERROR: invalid batch/length settings." >&2
    exit 1
fi

if (( NUM_INSTANCES == 1 )); then
    INSTANCE_SUFFIX=""
else
    INSTANCE_SUFFIX="_node$(printf '%05d' "${INSTANCE_RANK}")-of-$(printf '%05d' "${NUM_INSTANCES}")"
fi

# In pre-sharded mode, each instance receives its own balanced contiguous
# JSONL shard and divides it among local workers. Otherwise every instance
# reads the full JSONL and global ranks divide it without overlap.
if [[ "${PRE_SHARDED_INPUT}" == "1" ]]; then
    BASE_LOCAL_SAMPLES=$(( SFT_SAMPLES / NUM_INSTANCES ))
    EXTRA_LOCAL_SAMPLES=$(( SFT_SAMPLES % NUM_INSTANCES ))
    LOCAL_SAMPLES="${BASE_LOCAL_SAMPLES}"
    if (( INSTANCE_RANK < EXTRA_LOCAL_SAMPLES )); then
        LOCAL_SAMPLES=$(( LOCAL_SAMPLES + 1 ))
    fi
    GLOBAL_WORLD_SIZE="${NUM_WORKERS}"
    GLOBAL_RANK_START=0
    GLOBAL_RANK_END=$(( NUM_WORKERS - 1 ))
    PROMPT_FILENAME="openthoughts3_${SFT_SAMPLES}${INSTANCE_SUFFIX}.jsonl"
elif [[ "${PRE_SHARDED_INPUT}" == "0" ]]; then
    GLOBAL_WORLD_SIZE=$(( NUM_INSTANCES * NUM_WORKERS ))
    GLOBAL_RANK_START=$(( INSTANCE_RANK * NUM_WORKERS ))
    GLOBAL_RANK_END=$(( GLOBAL_RANK_START + NUM_WORKERS - 1 ))
    LOCAL_SAMPLES=0
    for (( global_rank=GLOBAL_RANK_START; global_rank<=GLOBAL_RANK_END; global_rank++ )); do
        if (( global_rank < SFT_SAMPLES )); then
            LOCAL_SAMPLES=$(( LOCAL_SAMPLES + (SFT_SAMPLES - 1 - global_rank) / GLOBAL_WORLD_SIZE + 1 ))
        fi
    done
    PROMPT_FILENAME="openthoughts3_${SFT_SAMPLES}.jsonl"
else
    echo "ERROR: PRE_SHARDED_INPUT must be 0 or 1." >&2
    exit 1
fi
if (( LOCAL_SAMPLES == 0 )); then
    echo "ERROR: this instance was assigned no samples." >&2
    exit 1
fi

PROMPT_DIR="${WORK_ROOT}/prompts"
MODEL_DIR="${WORK_ROOT}/models/Qwen3-8B"
PARTS_DIR="${WORK_ROOT}/sft_data/qwen3-8b_${SFT_SAMPLES}_${EXECUTION_TAG}${INSTANCE_SUFFIX}_arrow"
CHECKPOINT_DIR="${WORK_ROOT}/checkpoints/qwen3-8b_${SFT_SAMPLES}_${EXECUTION_TAG}${INSTANCE_SUFFIX}"
LOG_DIR="${WORK_ROOT}/logs/qwen3-8b_${SFT_SAMPLES}_${EXECUTION_TAG}${INSTANCE_SUFFIX}"
PROMPT_FILE="${PROMPT_DIR}/${PROMPT_FILENAME}"
FINAL_PARQUET="${WORK_ROOT}/sft_data/openthoughts3_${SFT_SAMPLES}_qwen3-8b${INSTANCE_SUFFIX}.parquet"
PRECOMPILE_OUTPUT="${WORK_ROOT}/precompile/qwen3-8b-${EXECUTION_TAG}"
PRECOMPILE_MARKER="${NEURON_COMPILED_ARTIFACTS}/_LIGHTNING_OPD_COMPILE_SUCCESS"

mkdir -p \
    "${HF_HOME}" \
    "${VLLM_CACHE_ROOT}" \
    "${NEURON_COMPILE_CACHE_URL}" \
    "${NEURON_COMPILED_ARTIFACTS}" \
    "${PROMPT_DIR}" \
    "${MODEL_DIR}" \
    "${PARTS_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${LOG_DIR}" \
    "${PRECOMPILE_OUTPUT}"

AVAILABLE_KIB="$(df -Pk "${WORK_ROOT}" | awk 'NR == 2 {print $4}')"
AVAILABLE_GIB=$(( AVAILABLE_KIB / 1024 / 1024 ))
if (( AVAILABLE_GIB < 300 )); then
    echo "WARNING: only ${AVAILABLE_GIB} GiB is free under ${WORK_ROOT}." >&2
    echo "A full long-reasoning curation run may need several hundred GiB." >&2
fi

echo "=== Lightning OPD paper SFT curation (NxD vLLM) ==="
echo "Instance:          ${INSTANCE_TYPE:-unknown}"
echo "Teacher:           ${TEACHER_MODEL}"
echo "Dataset:           ${HF_DATASET}"
echo "Prompts:           ${SFT_SAMPLES} (seed=${SEED})"
echo "Generation:        max_tokens=${MAX_TOKENS}, temperature=${TEMPERATURE}, top_p=${TOP_P}"
echo "Cluster:           instance $(( INSTANCE_RANK + 1 ))/${NUM_INSTANCES}"
echo "Topology:          ${NUM_WORKERS} local replicas x TP=${TP_SIZE} (${NUM_WORKERS} Trainium2 chips)"
echo "Input mode:        $([[ "${PRE_SHARDED_INPUT}" == "1" ]] && echo pre-sharded || echo global)"
echo "Worker ranks:      ${GLOBAL_RANK_START}-${GLOBAL_RANK_END} of ${GLOBAL_WORLD_SIZE}"
echo "Local prompts:     ${LOCAL_SAMPLES}"
echo "Per-replica batch: outer=${BATCH_SIZE}, compiled=${MAX_NUM_SEQS}"
echo "Work root:         ${WORK_ROOT}"
echo "Final parquet:     ${FINAL_PARQUET}"
echo "===================================================="

validate_prompts() {
    python - "${PROMPT_FILE}" "${LOCAL_SAMPLES}" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
count = 0
with open(path, encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        row = json.loads(line)
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise SystemExit(f"{path}:{line_number}: prompt must be a non-empty list")
        count += 1
if count != expected:
    raise SystemExit(f"{path}: found {count} prompts, expected {expected}")
print(f"Prompt validation passed: {count} rows")
PY
}

validate_parquet() {
    python - "${FINAL_PARQUET}" "$(( LOCAL_SAMPLES * NUM_RESPONSES ))" <<'PY'
import sys
import pyarrow.parquet as pq

path, expected = sys.argv[1], int(sys.argv[2])
parquet = pq.ParquetFile(path)
columns = set(parquet.schema_arrow.names)
missing = {"messages", "tokens"} - columns
if missing:
    raise SystemExit(f"{path}: missing columns {sorted(missing)}")
actual = parquet.metadata.num_rows
if actual != expected:
    raise SystemExit(f"{path}: found {actual} rows, expected {expected}")
print(f"Parquet validation passed: {actual} rows; columns={sorted(columns)}")
PY
}

if [[ -f "${FINAL_PARQUET}" ]] && validate_parquet; then
    echo ">>> Final paper SFT dataset is already complete."
    exit 0
fi

if [[ -f "${PROMPT_FILE}" ]] && validate_prompts; then
    echo ">>> Reusing prepared prompts."
elif [[ "${PRE_SHARDED_INPUT}" == "1" ]]; then
    echo "ERROR: required pre-sharded prompt file is missing or invalid:" >&2
    echo "  ${PROMPT_FILE}" >&2
    echo "Download the matching node shard before launching." >&2
    exit 1
else
    PROMPT_TMP="${PROMPT_FILE}.tmp.$$"
    echo ">>> Downloading OpenThoughts3 and preparing ${SFT_SAMPLES} prompts."
    python scripts/prepare_sft_prompts.py \
        --hf-dataset "${HF_DATASET}" \
        --output "${PROMPT_TMP}" \
        --num-samples "${SFT_SAMPLES}" \
        --seed "${SEED}"
    mv "${PROMPT_TMP}" "${PROMPT_FILE}"
    validate_prompts
fi

PROMPT_SHA256="$(sha256sum "${PROMPT_FILE}" | awk '{print $1}')"
echo "Prompt SHA-256: ${PROMPT_SHA256}"
if [[ -n "${EXPECTED_PROMPT_SHA256:-}" && "${PROMPT_SHA256}" != "${EXPECTED_PROMPT_SHA256}" ]]; then
    echo "ERROR: prompt SHA-256 does not match EXPECTED_PROMPT_SHA256." >&2
    exit 1
fi
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    echo ">>> PREPARE_ONLY=1; prompt preparation is complete."
    exit 0
fi

echo ">>> Downloading ${TEACHER_MODEL} to a local path."
# A local checkpoint is required for Qwen3 when NxDI shard-on-load is enabled
# because Qwen3-8B uses tied word embeddings.
if command -v hf >/dev/null 2>&1; then
    hf download "${TEACHER_MODEL}" --local-dir "${MODEL_DIR}"
else
    huggingface-cli download "${TEACHER_MODEL}" --local-dir "${MODEL_DIR}"
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: model download did not create ${MODEL_DIR}/config.json." >&2
    exit 1
fi

PIPELINE_ARGS=(
    --model "${MODEL_DIR}"
    --input "${PROMPT_FILE}"
    --output-dir "${PARTS_DIR}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-tokens "${MAX_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --num-responses "${NUM_RESPONSES}"
    --batch-size "${BATCH_SIZE}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --num-gpu-blocks-override "${MAX_NUM_SEQS}"
    --dtype bfloat16
    --enable-thinking
    --download-dir "${HF_HOME}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
)

if [[ "${PRECOMPILE}" == "1" && ! -f "${PRECOMPILE_MARKER}" ]]; then
    echo ">>> Compiling Qwen3-8B once on logical NeuronCores 0-$(( TP_SIZE - 1 ))."
    echo "    All replicas will reuse ${NEURON_COMPILED_ARTIFACTS}."
    NEURON_RT_VISIBLE_CORES="0-$(( TP_SIZE - 1 ))" \
        python data_curation/pipeline.py \
            "${PIPELINE_ARGS[@]}" \
            --output-dir "${PRECOMPILE_OUTPUT}" \
            --rank 0 \
            --world-size 1 \
            --num-samples 0
    touch "${PRECOMPILE_MARKER}"
fi

PIDS=()
cleanup_workers() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup_workers INT TERM

echo ">>> Launching missing curation shards."
for (( local_rank=0; local_rank<NUM_WORKERS; local_rank++ )); do
    rank=$(( GLOBAL_RANK_START + local_rank ))
    RANK_DIR="${PARTS_DIR}/rank$(printf '%05d' "${rank}")"
    if [[ -f "${RANK_DIR}/_SUCCESS" ]]; then
        echo "    rank ${rank}: already complete"
        continue
    fi

    CORE_START=$(( local_rank * TP_SIZE ))
    CORE_END=$(( CORE_START + TP_SIZE - 1 ))
    RANK_LOG="${LOG_DIR}/rank$(printf '%05d' "${rank}").log"
    echo "    rank ${rank}: cores ${CORE_START}-${CORE_END}; log ${RANK_LOG}"

    NEURON_RT_VISIBLE_CORES="${CORE_START}-${CORE_END}" \
        python data_curation/pipeline.py \
            "${PIPELINE_ARGS[@]}" \
            --rank "${rank}" \
            --world-size "${GLOBAL_WORLD_SIZE}" \
            >"${RANK_LOG}" 2>&1 &
    PIDS+=("$!")
done

if (( ${#PIDS[@]} > 0 )); then
    echo ">>> Waiting for ${#PIDS[@]} workers. Follow progress with:"
    echo "    tail -F ${LOG_DIR}/rank$(printf '%05d' "${GLOBAL_RANK_START}").log"
    REMAINING=${#PIDS[@]}
    FAILED=0
    set +e
    while (( REMAINING > 0 )); do
        wait -n
        STATUS=$?
        if (( STATUS != 0 )); then
            FAILED=1
            break
        fi
        REMAINING=$(( REMAINING - 1 ))
        echo "    worker completed; ${REMAINING} still running"
    done
    set -e

    if (( FAILED != 0 )); then
        echo "ERROR: a curation worker failed; stopping remaining workers." >&2
        cleanup_workers
        wait || true
        echo "Inspect logs under ${LOG_DIR}; the next run resumes incomplete ranks." >&2
        exit 1
    fi
fi
trap - INT TERM

for (( local_rank=0; local_rank<NUM_WORKERS; local_rank++ )); do
    rank=$(( GLOBAL_RANK_START + local_rank ))
    SUCCESS_MARKER="${PARTS_DIR}/rank$(printf '%05d' "${rank}")/_SUCCESS"
    if [[ ! -f "${SUCCESS_MARKER}" ]]; then
        echo "ERROR: missing completion marker ${SUCCESS_MARKER}." >&2
        exit 1
    fi
done

echo ">>> Merging Arrow shards into the paper SFT parquet."
python data_curation/merge.py \
    --input-dir "${PARTS_DIR}" \
    --output "${FINAL_PARQUET}" \
    --max-tokens "${MAX_TOKENS}"
validate_parquet

echo
echo "=== Curation complete ==="
echo "SFT dataset: ${FINAL_PARQUET}"
echo "Raw Arrow shards and checkpoints were retained for audit/resume."
