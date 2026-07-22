#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3-8B SFT-curation smoke test for the direct vLLM Neuron beta plugin on
# one trn2.3xlarge. This uses the Qwen3 implementation from a local
# vllm-neuron source checkout, not the legacy NxD Inference integration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 5 )); then
    echo "ERROR: Bash 5 or newer is required." >&2
    exit 1
fi

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

if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "ERROR: neuron-ls is unavailable. Run on a Neuron 2.31+ DLAMI." >&2
    exit 1
fi
neuron-ls

# Activate the direct vLLM 0.21 beta environment unless the caller already
# selected a virtual environment. Override VLLM_VENV for a non-DLAMI layout.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -n "${VLLM_VENV:-}" ]]; then
        if [[ ! -f "${VLLM_VENV}/bin/activate" ]]; then
            echo "ERROR: VLLM_VENV has no bin/activate: ${VLLM_VENV}" >&2
            exit 1
        fi
        # shellcheck disable=SC1090
        source "${VLLM_VENV}/bin/activate"
    else
        for candidate in \
            /opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0 \
            "${HOME}/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0"; do
            if [[ -f "${candidate}/bin/activate" ]]; then
                # shellcheck disable=SC1090
                source "${candidate}/bin/activate"
                break
            fi
        done
    fi
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: the vLLM Neuron beta virtual environment was not found." >&2
    echo "Set VLLM_VENV to the Neuron 2.31 vLLM 0.21 environment." >&2
    exit 1
fi

# Locate the Qwen3-enabled source checkout. The first path is the deployment
# location used during Qwen3 onboarding; VLLM_NEURON_SRC always takes priority.
if [[ -z "${VLLM_NEURON_SRC:-}" ]]; then
    for candidate in \
        "${HOME}/src/vllm-neuron-qwen3" \
        "${HOME}/src/vllm-neuron" \
        "${REPO_ROOT}/../vllm-neuron" \
        "${HOME}/vllm-neuron"; do
        if [[ -f "${candidate}/pyproject.toml" ]]; then
            VLLM_NEURON_SRC="${candidate}"
            break
        fi
    done
fi
if [[ -z "${VLLM_NEURON_SRC:-}" || ! -f "${VLLM_NEURON_SRC}/pyproject.toml" ]]; then
    echo "ERROR: cannot find the local vllm-neuron checkout." >&2
    echo "Set VLLM_NEURON_SRC=/absolute/path/to/vllm-neuron." >&2
    exit 1
fi
VLLM_NEURON_SRC="$(cd "${VLLM_NEURON_SRC}" && pwd)"
export VLLM_NEURON_SRC

if [[ ! -f "${VLLM_NEURON_SRC}/vllm_neuron/model/qwen3/model.py" ]] || \
   ! grep -q 'Qwen3ForCausalLM' "${VLLM_NEURON_SRC}/vllm_neuron/model/registry.py"; then
    echo "ERROR: ${VLLM_NEURON_SRC} does not contain the Qwen3 beta port." >&2
    exit 1
fi

plugin_uses_source() {
    python - "${VLLM_NEURON_SRC}" <<'PY'
import inspect
import pathlib
import sys

try:
    import vllm_neuron
    from vllm_neuron.model.qwen3 import Qwen3ForCausalLM
except Exception as error:
    print(f"Plugin import check failed: {error}", file=sys.stderr)
    raise SystemExit(1)

root = pathlib.Path(sys.argv[1]).resolve()
module_path = pathlib.Path(inspect.getfile(vllm_neuron)).resolve()
model_path = pathlib.Path(inspect.getfile(Qwen3ForCausalLM)).resolve()
if root not in module_path.parents or root not in model_path.parents:
    print(f"Installed plugin is not sourced from {root}: {module_path}", file=sys.stderr)
    raise SystemExit(1)
PY
}

INSTALL_PLUGIN="${INSTALL_PLUGIN:-auto}"
case "${INSTALL_PLUGIN}" in
    1)
        python -m pip install \
            --extra-index-url=https://pip.repos.neuron.amazonaws.com \
            -e "${VLLM_NEURON_SRC}"
        ;;
    auto)
        if ! plugin_uses_source; then
            echo ">>> Installing the local Qwen3-enabled vllm-neuron checkout."
            python -m pip install \
                --extra-index-url=https://pip.repos.neuron.amazonaws.com \
                -e "${VLLM_NEURON_SRC}"
        fi
        ;;
    0)
        ;;
    *)
        echo "ERROR: INSTALL_PLUGIN must be auto, 0, or 1." >&2
        exit 1
        ;;
esac
if ! plugin_uses_source; then
    echo "ERROR: vllm-neuron is not installed from ${VLLM_NEURON_SRC}." >&2
    echo "Rerun with INSTALL_PLUGIN=1." >&2
    exit 1
fi

if ! python - <<'PY'
import huggingface_hub
import pandas
import pyarrow
import tqdm
PY
then
    python -m pip install huggingface_hub pandas pyarrow tqdm
fi

TEACHER_MODEL="Qwen/Qwen3-8B"
HF_DATASET="open-thoughts/OpenThoughts3-1.2M"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TP_SIZE="${TP_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEMPERATURE="0.7"
TOP_P="0.9"
NUM_RESPONSES=1

for positive_integer in \
    "${SMOKE_SAMPLES}" "${MAX_TOKENS}" "${MAX_MODEL_LEN}" \
    "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" "${TP_SIZE}" \
    "${BATCH_SIZE}"; do
    if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: counts, lengths, and parallel sizes must be positive integers." >&2
        exit 1
    fi
done
if (( TP_SIZE > 4 )); then
    echo "ERROR: TP_SIZE=${TP_SIZE} exceeds the four logical cores on trn2.3xlarge." >&2
    exit 1
fi
if (( MAX_MODEL_LEN <= MAX_TOKENS )); then
    echo "ERROR: MAX_MODEL_LEN must exceed MAX_TOKENS for prompt tokens." >&2
    exit 1
fi
case "${MAX_NUM_BATCHED_TOKENS}" in
    512|1024|2048|4096|8192) ;;
    *)
        echo "ERROR: MAX_NUM_BATCHED_TOKENS must be 512, 1024, 2048, 4096, or 8192." >&2
        echo "The beta plugin requires segmented prefill above a 16K context." >&2
        exit 1
        ;;
esac
if (( MAX_NUM_BATCHED_TOKENS >= MAX_MODEL_LEN )); then
    echo "ERROR: MAX_NUM_BATCHED_TOKENS must be below MAX_MODEL_LEN for segmented prefill." >&2
    exit 1
fi

# Reuse the writable NVMe workspace prepared during Qwen3 onboarding when it
# is exported as QWEN_WORK. Otherwise remain inside this repository by default.
WORK_ROOT="${WORK_ROOT:-${QWEN_WORK:-${REPO_ROOT}/data/trainium/beta}}"
export WORK_ROOT
export HF_HOME="${HF_HOME:-${WORK_ROOT}/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORK_ROOT}/cache/vllm}"
export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
export VLLM_NEURON_COMPILATION_TIMEOUT="${VLLM_NEURON_COMPILATION_TIMEOUT:-7200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-7200}"
export VLLM_NEURON_LOG_LEVEL="${VLLM_NEURON_LOG_LEVEL:-INFO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# trn2.3xlarge has no usable EFA device. Affinity is only an optimization.
export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXECUTION_TAG="tp${TP_SIZE}-seqs${MAX_NUM_SEQS}-len${MAX_MODEL_LEN}"
SMOKE_ROOT="${WORK_ROOT}/smoke"
RUN_DIR="${SMOKE_ROOT}/runs/beta-${EXECUTION_TAG}-${RUN_ID}"
PROMPT_FILE="${SMOKE_ROOT}/prompts/openthoughts3_first_${SMOKE_SAMPLES}.jsonl"
MODEL_DIR="${WORK_ROOT}/models/Qwen3-8B"
PARTS_DIR="${RUN_DIR}/arrow"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"
PRECOMPILE_OUTPUT="${RUN_DIR}/precompile"
FINAL_PARQUET="${RUN_DIR}/qwen3-8b-smoke.parquet"
REPORT_FILE="${RUN_DIR}/benchmark.json"

mkdir -p \
    "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${TMPDIR}" \
    "$(dirname "${PROMPT_FILE}")" "${MODEL_DIR}" \
    "${PARTS_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}" \
    "${PRECOMPILE_OUTPUT}"

PLUGIN_COMMIT="$(git -C "${VLLM_NEURON_SRC}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
python - "${VLLM_NEURON_SRC}" <<'PY'
import importlib.metadata
import inspect
import sys

import vllm
import vllm_neuron
from vllm.platforms import current_platform

print(f"Python: {sys.version.split()[0]}")
print(f"vLLM: {vllm.__version__}")
print(f"vllm-neuron: {importlib.metadata.version('vllm-neuron')}")
print(f"Plugin source: {inspect.getfile(vllm_neuron)}")
print(f"Platform: {current_platform.device_name}")
if current_platform.device_name != "neuron":
    raise SystemExit("vLLM did not select the Neuron platform")
PY

echo "=== Lightning OPD direct-beta Qwen3-8B smoke test ==="
echo "Instance:          ${INSTANCE_TYPE:-unknown}"
echo "Plugin source:     ${VLLM_NEURON_SRC} (${PLUGIN_COMMIT})"
echo "Prompts:           ${SMOKE_SAMPLES}"
echo "Generation:        max_tokens=${MAX_TOKENS}, max_model_len=${MAX_MODEL_LEN}"
echo "Sampling:          temperature=${TEMPERATURE}, top_p=${TOP_P}, thinking=on"
echo "Topology:          1 replica x TP=${TP_SIZE}, max_num_seqs=${MAX_NUM_SEQS}"
echo "Prefill segment:   ${MAX_NUM_BATCHED_TOKENS} tokens"
echo "Run directory:     ${RUN_DIR}"
echo "========================================================="

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
    echo ">>> Fetching ${SMOKE_SAMPLES} prompts from the Hugging Face Dataset Viewer."
    if ! python - "${HF_DATASET}" "${SMOKE_SAMPLES}" "${PROMPT_TMP}" <<'PY'
import json
import os
import sys
import urllib.parse
import urllib.request

from scripts.prepare_sft_prompts import extract_prompt

dataset_name, count, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base_url = "https://datasets-server.huggingface.co/rows"
headers = {"User-Agent": "Lightning-OPD-vLLM-Neuron-beta-smoke/1.0"}
if token := os.environ.get("HF_TOKEN"):
    headers["Authorization"] = f"Bearer {token}"
offset = 0
written = 0

with open(output, "w", encoding="utf-8") as handle:
    while written < count:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset_name,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": min(100, max(count - written, 10)),
            }
        )
        request = urllib.request.Request(f"{base_url}?{query}", headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        page = payload.get("rows", [])
        if not page:
            break
        for envelope in page:
            row = extract_prompt(envelope["row"])
            if row is None or not row["prompt"]:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written == count:
                break
        offset += len(page)
        total = payload.get("num_rows_total")
        if total is not None and offset >= total:
            break

if written != count:
    raise SystemExit(f"Only found {written} valid prompts; expected {count}")
print(f"Fetched {written} prompts")
PY
    then
        rm -f "${PROMPT_TMP}"
        exit 1
    fi
    mv "${PROMPT_TMP}" "${PROMPT_FILE}"
    validate_prompts
fi

echo ">>> Downloading ${TEACHER_MODEL} to ${MODEL_DIR}."
if command -v hf >/dev/null 2>&1; then
    hf download "${TEACHER_MODEL}" --local-dir "${MODEL_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${TEACHER_MODEL}" --local-dir "${MODEL_DIR}"
else
    echo "ERROR: neither hf nor huggingface-cli is available." >&2
    exit 1
fi

python - "${MODEL_DIR}" <<'PY'
import sys
from transformers import AutoConfig

config = AutoConfig.from_pretrained(sys.argv[1])
if config.architectures != ["Qwen3ForCausalLM"]:
    raise SystemExit(f"Unexpected model architecture: {config.architectures}")
if getattr(config, "quantization_config", None) is not None:
    raise SystemExit("The beta Qwen3 port supports only unquantized BF16 checkpoints")
print(f"Model config passed: {config.architectures[0]}")
PY

# Qwen3's 18,432-token context is above the beta plugin's 16K single-shot
# prefill limit, so use its supported 2,048-token segmented-prefill path. The
# generation cap, context cap, concurrency, sampling, and TP match the NxD smoke.
ADDITIONAL_CONFIG_JSON="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${MAX_NUM_BATCHED_TOKENS}],\"num_seqs_buckets\":[${MAX_NUM_SEQS}],\"on_device_sampling_config\":{\"all_greedy\":false}}}"

PIPELINE_ARGS=(
    --model "${MODEL_DIR}"
    --input "${PROMPT_FILE}"
    --output-dir "${PARTS_DIR}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-tokens "${MAX_TOKENS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --num-responses "${NUM_RESPONSES}"
    --batch-size "${BATCH_SIZE}"
    --dtype bfloat16
    --enable-thinking
    --download-dir "${HF_HOME}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --additional-config-json "${ADDITIONAL_CONFIG_JSON}"
)

CORE_END=$(( TP_SIZE - 1 ))
echo ">>> Preparing/compiling the beta engine on logical cores 0-${CORE_END}."
PREPARE_START="$(date +%s)"
NEURON_RT_VISIBLE_CORES="0-${CORE_END}" \
    python data_curation/pipeline.py \
        "${PIPELINE_ARGS[@]}" \
        --output-dir "${PRECOMPILE_OUTPUT}" \
        --rank 0 \
        --world-size 1 \
        --num-samples 0 \
        2>&1 | tee "${LOG_DIR}/prepare.log"
PREPARE_SECONDS=$(( $(date +%s) - PREPARE_START ))

echo ">>> Generating ${SMOKE_SAMPLES} paper-configured responses."
GENERATION_START="$(date +%s)"
NEURON_RT_VISIBLE_CORES="0-${CORE_END}" \
    python data_curation/pipeline.py \
        "${PIPELINE_ARGS[@]}" \
        --rank 0 \
        --world-size 1 \
        2>&1 | tee "${LOG_DIR}/generation.log"
GENERATION_SECONDS=$(( $(date +%s) - GENERATION_START ))

if [[ ! -f "${PARTS_DIR}/_SUCCESS" ]]; then
    echo "ERROR: generation completed without ${PARTS_DIR}/_SUCCESS." >&2
    exit 1
fi

echo ">>> Merging and validating smoke output."
python data_curation/merge.py \
    --input-dir "${PARTS_DIR}" \
    --output "${FINAL_PARQUET}" \
    --max-tokens "${MAX_TOKENS}"

python - \
    "${FINAL_PARQUET}" "${REPORT_FILE}" "${INSTANCE_TYPE:-unknown}" \
    "${TP_SIZE}" "${MAX_NUM_SEQS}" "${SMOKE_SAMPLES}" "${MAX_TOKENS}" \
    "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" "${PREPARE_SECONDS}" \
    "${GENERATION_SECONDS}" "${PLUGIN_COMMIT}" "${VLLM_NEURON_SRC}" <<'PY'
import json
import sys

import pyarrow.parquet as pq

(
    parquet_path,
    report_path,
    instance_type,
    tp_size,
    max_num_seqs,
    expected_rows,
    max_tokens,
    max_model_len,
    max_num_batched_tokens,
    prepare_seconds,
    generation_seconds,
    plugin_commit,
    plugin_source,
) = sys.argv[1:]

tokens = [
    int(value)
    for value in pq.read_table(parquet_path, columns=["tokens"])
    .column("tokens")
    .to_pylist()
]
expected_rows = int(expected_rows)
if len(tokens) != expected_rows:
    raise SystemExit(f"{parquet_path}: found {len(tokens)} rows, expected {expected_rows}")
if any(count <= 0 for count in tokens):
    raise SystemExit(f"{parquet_path}: one or more completions are empty")

generation_seconds = int(generation_seconds)
report = {
    "backend": "vllm-neuron-direct-beta",
    "instance_type": instance_type,
    "plugin_source": plugin_source,
    "plugin_commit": plugin_commit,
    "tensor_parallel_size": int(tp_size),
    "replicas": 1,
    "max_num_seqs": int(max_num_seqs),
    "max_num_batched_tokens": int(max_num_batched_tokens),
    "samples": len(tokens),
    "generated_tokens": sum(tokens),
    "average_generated_tokens": sum(tokens) / len(tokens),
    "max_tokens": int(max_tokens),
    "max_model_len": int(max_model_len),
    "temperature": 0.7,
    "top_p": 0.9,
    "thinking": True,
    "engine_prepare_seconds": int(prepare_seconds),
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

echo
echo "Direct-beta smoke test passed."
echo "Result: ${FINAL_PARQUET}"
echo "Report: ${REPORT_FILE}"
