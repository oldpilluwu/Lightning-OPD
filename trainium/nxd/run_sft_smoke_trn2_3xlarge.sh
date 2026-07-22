#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-chip Qwen3-8B SFT-curation smoke test for trn2.3xlarge.
#
# The production sequence length and sampling settings are retained, but only
# a few OpenThoughts3 prompts are streamed. The default is one TP=4 replica.
# Set TP_SIZE=2 to test two independent TP=2 replicas on the same chip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 5 )); then
    echo "ERROR: Bash 5 or newer is required." >&2
    exit 1
fi

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/data/trainium/nxd}"
export WORK_ROOT
export HF_HOME="${HF_HOME:-${WORK_ROOT}/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORK_ROOT}/cache/vllm}"
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-${WORK_ROOT}/cache/neuron-compile-cache}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_env.sh"

EXPECTED_INSTANCE_TYPE="trn2.3xlarge"
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

TEACHER_MODEL="Qwen/Qwen3-8B"
HF_DATASET="open-thoughts/OpenThoughts3-1.2M"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
TEMPERATURE="0.7"
TOP_P="0.9"
NUM_RESPONSES=1
TP_SIZE="${TP_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Under LNC=2 a single Trainium2 chip exposes four logical NeuronCores.
# Keep eight scheduled sequences across the chip for an apples-to-apples test:
#   TP=4: 1 replica x 8 sequences
#   TP=2: 2 replicas x 4 sequences
#   TP=1: 4 replicas x 2 sequences (allowed as an experimental fit check)
case "${TP_SIZE}" in
    4)
        DEFAULT_NUM_WORKERS=1
        DEFAULT_MAX_NUM_SEQS=8
        ;;
    2)
        DEFAULT_NUM_WORKERS=2
        DEFAULT_MAX_NUM_SEQS=4
        ;;
    1)
        DEFAULT_NUM_WORKERS=4
        DEFAULT_MAX_NUM_SEQS=2
        ;;
    *)
        echo "ERROR: TP_SIZE must be 1, 2, or 4 on one LNC2 Trainium2 chip." >&2
        exit 1
        ;;
esac

NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
TOTAL_LOGICAL_CORES=4

for positive_integer in \
    "${SMOKE_SAMPLES}" "${MAX_TOKENS}" "${MAX_MODEL_LEN}" \
    "${BATCH_SIZE}" "${NUM_WORKERS}" "${MAX_NUM_SEQS}"; do
    if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: smoke counts, batch sizes, and lengths must be positive integers." >&2
        exit 1
    fi
done
if (( NUM_WORKERS * TP_SIZE > TOTAL_LOGICAL_CORES )); then
    echo "ERROR: ${NUM_WORKERS} replicas x TP=${TP_SIZE} exceeds four LNC2 cores." >&2
    exit 1
fi
if (( SMOKE_SAMPLES < NUM_WORKERS )); then
    echo "ERROR: SMOKE_SAMPLES must be at least NUM_WORKERS (${NUM_WORKERS})." >&2
    exit 1
fi
if (( MAX_MODEL_LEN <= MAX_TOKENS )); then
    echo "ERROR: MAX_MODEL_LEN must exceed MAX_TOKENS to leave room for the prompt." >&2
    exit 1
fi

EXECUTION_TAG="tp${TP_SIZE}-workers${NUM_WORKERS}-seqs${MAX_NUM_SEQS}-len${MAX_MODEL_LEN}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ROOT="${WORK_ROOT}/smoke"
RUN_DIR="${SMOKE_ROOT}/runs/${EXECUTION_TAG}-${RUN_ID}"
PROMPT_FILE="${SMOKE_ROOT}/prompts/openthoughts3_first_${SMOKE_SAMPLES}.jsonl"
MODEL_DIR="${WORK_ROOT}/models/Qwen3-8B"
PARTS_DIR="${RUN_DIR}/arrow"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
FINAL_PARQUET="${RUN_DIR}/qwen3-8b-smoke.parquet"
REPORT_FILE="${RUN_DIR}/benchmark.json"
PRECOMPILE_OUTPUT="${RUN_DIR}/precompile"
export NEURON_COMPILED_ARTIFACTS="${NEURON_COMPILED_ARTIFACTS:-${WORK_ROOT}/compiled/qwen3-8b-${EXECUTION_TAG}}"
PRECOMPILE_MARKER="${NEURON_COMPILED_ARTIFACTS}/_LIGHTNING_OPD_COMPILE_SUCCESS"

mkdir -p \
    "${HF_HOME}" \
    "${VLLM_CACHE_ROOT}" \
    "${NEURON_COMPILE_CACHE_URL}" \
    "${NEURON_COMPILED_ARTIFACTS}" \
    "$(dirname "${PROMPT_FILE}")" \
    "${MODEL_DIR}" \
    "${PARTS_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${LOG_DIR}" \
    "${PRECOMPILE_OUTPUT}"

echo "=== Lightning OPD Qwen3-8B one-chip smoke test ==="
echo "Instance:          ${INSTANCE_TYPE:-unknown}"
echo "Prompts:           ${SMOKE_SAMPLES}"
echo "Generation:        max_tokens=${MAX_TOKENS}, max_model_len=${MAX_MODEL_LEN}"
echo "Sampling:          temperature=${TEMPERATURE}, top_p=${TOP_P}, thinking=on"
echo "Topology:          ${NUM_WORKERS} replicas x TP=${TP_SIZE}"
echo "Compiled slots:    ${MAX_NUM_SEQS} per replica ($(( NUM_WORKERS * MAX_NUM_SEQS )) total)"
echo "Run directory:     ${RUN_DIR}"
echo "==================================================="

validate_prompts() {
    python - "${PROMPT_FILE}" "${SMOKE_SAMPLES}" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle]
if len(rows) != expected:
    raise SystemExit(f"{path}: found {len(rows)} prompts, expected {expected}")
for index, row in enumerate(rows):
    if not isinstance(row.get("prompt"), list) or not row["prompt"]:
        raise SystemExit(f"{path}: row {index} has no chat prompt")
print(f"Prompt validation passed: {len(rows)} rows")
PY
}

if [[ -f "${PROMPT_FILE}" ]] && validate_prompts; then
    echo ">>> Reusing smoke prompts."
else
    PROMPT_TMP="${PROMPT_FILE}.tmp.$$"
    echo ">>> Streaming ${SMOKE_SAMPLES} valid prompts from ${HF_DATASET}."
    python - "${HF_DATASET}" "${SMOKE_SAMPLES}" "${PROMPT_TMP}" <<'PY'
import json
import sys

from datasets import load_dataset
from scripts.prepare_sft_prompts import extract_prompt

dataset_name, count, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dataset = load_dataset(dataset_name, split="train", streaming=True)
written = 0
with open(output, "w", encoding="utf-8") as handle:
    for sample in dataset:
        row = extract_prompt(sample)
        if row is None or not row["prompt"]:
            continue
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
        if written == count:
            break
if written != count:
    raise SystemExit(f"Only found {written} valid prompts; expected {count}")
print(f"Streamed {written} prompts")
PY
    mv "${PROMPT_TMP}" "${PROMPT_FILE}"
    validate_prompts
fi

echo ">>> Downloading ${TEACHER_MODEL} to a local path when needed."
# Qwen3 has tied embeddings, so the NxDI vLLM guide requires a local model path.
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

COMPILE_CACHE_REUSED=0
COMPILE_START="$(date +%s)"
if [[ -f "${PRECOMPILE_MARKER}" ]]; then
    COMPILE_CACHE_REUSED=1
    echo ">>> Reusing compiled artifacts from ${NEURON_COMPILED_ARTIFACTS}."
else
    echo ">>> Compiling TP=${TP_SIZE} on logical cores 0-$(( TP_SIZE - 1 ))."
    NEURON_RT_VISIBLE_CORES="0-$(( TP_SIZE - 1 ))" \
        python data_curation/pipeline.py \
            "${PIPELINE_ARGS[@]}" \
            --output-dir "${PRECOMPILE_OUTPUT}" \
            --rank 0 \
            --world-size 1 \
            --num-samples 0
    touch "${PRECOMPILE_MARKER}"
fi
COMPILE_SECONDS=$(( $(date +%s) - COMPILE_START ))

PIDS=()
cleanup_workers() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup_workers INT TERM

echo ">>> Running ${SMOKE_SAMPLES} prompts across ${NUM_WORKERS} worker(s)."
GENERATION_START="$(date +%s)"
for (( rank=0; rank<NUM_WORKERS; rank++ )); do
    CORE_START=$(( rank * TP_SIZE ))
    CORE_END=$(( CORE_START + TP_SIZE - 1 ))
    RANK_LOG="${LOG_DIR}/rank$(printf '%05d' "${rank}").log"
    echo "    rank ${rank}: cores ${CORE_START}-${CORE_END}; log ${RANK_LOG}"
    NEURON_RT_VISIBLE_CORES="${CORE_START}-${CORE_END}" \
        python data_curation/pipeline.py \
            "${PIPELINE_ARGS[@]}" \
            --rank "${rank}" \
            --world-size "${NUM_WORKERS}" \
            >"${RANK_LOG}" 2>&1 &
    PIDS+=("$!")
done

REMAINING=${#PIDS[@]}
set +e
while (( REMAINING > 0 )); do
    wait -n
    STATUS=$?
    if (( STATUS != 0 )); then
        set -e
        echo "ERROR: a smoke worker failed; stopping the remaining workers." >&2
        cleanup_workers
        wait || true
        echo "Logs: ${LOG_DIR}" >&2
        exit 1
    fi
    REMAINING=$(( REMAINING - 1 ))
    echo "    worker completed; ${REMAINING} still running"
done
set -e
trap - INT TERM
GENERATION_SECONDS=$(( $(date +%s) - GENERATION_START ))

for (( rank=0; rank<NUM_WORKERS; rank++ )); do
    if (( NUM_WORKERS == 1 )); then
        SUCCESS_MARKER="${PARTS_DIR}/_SUCCESS"
    else
        SUCCESS_MARKER="${PARTS_DIR}/rank$(printf '%05d' "${rank}")/_SUCCESS"
    fi
    if [[ ! -f "${SUCCESS_MARKER}" ]]; then
        echo "ERROR: missing worker completion marker ${SUCCESS_MARKER}." >&2
        exit 1
    fi
done

echo ">>> Merging and validating smoke output."
python data_curation/merge.py \
    --input-dir "${PARTS_DIR}" \
    --output "${FINAL_PARQUET}" \
    --max-tokens "${MAX_TOKENS}"

python - \
    "${FINAL_PARQUET}" "${REPORT_FILE}" "${INSTANCE_TYPE:-unknown}" \
    "${TP_SIZE}" "${NUM_WORKERS}" "${MAX_NUM_SEQS}" "${SMOKE_SAMPLES}" \
    "${MAX_TOKENS}" "${MAX_MODEL_LEN}" "${COMPILE_SECONDS}" \
    "${GENERATION_SECONDS}" "${COMPILE_CACHE_REUSED}" <<'PY'
import json
import sys

import pyarrow.parquet as pq

(
    parquet_path,
    report_path,
    instance_type,
    tp_size,
    workers,
    max_num_seqs,
    expected_rows,
    max_tokens,
    max_model_len,
    compile_seconds,
    generation_seconds,
    compile_cache_reused,
) = sys.argv[1:]

table = pq.read_table(parquet_path, columns=["tokens"])
tokens = [int(value) for value in table.column("tokens").to_pylist()]
expected_rows = int(expected_rows)
if len(tokens) != expected_rows:
    raise SystemExit(f"{parquet_path}: found {len(tokens)} rows, expected {expected_rows}")

generation_seconds = int(generation_seconds)
report = {
    "instance_type": instance_type,
    "tensor_parallel_size": int(tp_size),
    "replicas": int(workers),
    "max_num_seqs_per_replica": int(max_num_seqs),
    "samples": len(tokens),
    "generated_tokens": sum(tokens),
    "average_generated_tokens": sum(tokens) / len(tokens),
    "max_tokens": int(max_tokens),
    "max_model_len": int(max_model_len),
    "compile_cache_reused": bool(int(compile_cache_reused)),
    "compile_seconds": int(compile_seconds),
    "generation_seconds": generation_seconds,
    "generated_tokens_per_second": sum(tokens) / max(generation_seconds, 1),
    "parquet": parquet_path,
}
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")

print("=== Smoke benchmark result ===")
for key, value in report.items():
    print(f"{key}: {value}")
PY

PRODUCTION_WORKERS=$(( NUM_WORKERS * 16 ))
echo
echo "Smoke test passed. Result: ${REPORT_FILE}"
echo "Equivalent trn2.48xlarge topology command:"
echo "TP_SIZE=${TP_SIZE} NUM_WORKERS=${PRODUCTION_WORKERS} MAX_NUM_SEQS=${MAX_NUM_SEQS} BATCH_SIZE=${BATCH_SIZE} \\"
echo "  bash trainium/nxd/run_sft_curation_trn2_48xlarge.sh"
