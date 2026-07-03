#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Single-GPU Qwen3-4B student / Qwen3-8B teacher Lightning-OPD pipeline.
#
# Stages (all under RUN_DIR, each skipped once complete):
#   1. sample 5k SFT prompts (OpenThoughts3) + 2k OPD prompts (DAPO-Math-17k)
#   2. generate SFT data with the teacher (vLLM, batch-size 128)
#   3. SFT in 5-step chunks; after each chunk a fixed teacher-generated OPD
#      probe batch is scored against the checkpoint and logged to
#      RUN_DIR/metrics/sft_checkpoint_opd_probe_metrics.{jsonl,csv}:
#        probe/teacher_nll                     teacher NLL on the probe batch
#        probe/student_nll                     student NLL on the same batch
#        probe/kl_mc_teacher_to_student        MC estimate of KL(teacher||student)
#        probe/policy_drift_prev_to_current_mc mean logprob shift vs prev ckpt
#        probe/policy_drift_abs_logprob_delta  mean |logprob shift| vs prev ckpt
#      The previous checkpoint is deleted after scoring; the latest one is
#      kept because it is required to resume training.
#   4. collect student rollouts on the OPD prompts
#   5. precompute teacher logprobs (sglang server, started/stopped here)
#   6. Lightning OPD training on 1 GPU
#
# Resumability:
#   - Stage completion markers live in RUN_DIR/.done.
#   - SFT resumes from the newest retained checkpoint; per-step metrics that
#     already exist are skipped.
#   - vLLM curation stages checkpoint per batch and resume mid-dataset.
#
# Environments (README setup, managed automatically when USE_CONDA=1):
#   - conda env "curation"     prompts + vLLM generation (stages 1, 2, 4)
#   - conda env "llamafactory" SFT training + checkpoint metrics (stage 3)
#   - OPD stages (5, 6) run in the CURRENT environment by default because
#     they need sglang + Megatron, which the repo provides via its docker
#     container (bash run_docker.sh). Set OPD_ENV to a conda env name if you
#     have those dependencies in conda instead.
#   Each env is created and populated once (marker-gated), activated before
#   its stage and deactivated afterwards. Set USE_CONDA=0 to run everything
#   in the current environment (e.g. inside a single mega-env or container).
#   If conda itself is missing, Miniconda is auto-installed to
#   CONDA_INSTALL_DIR (default ~/miniconda3); set AUTO_INSTALL_CONDA=0 to
#   disable and error out instead.
#
# Example:
#   bash scripts/run_single_gpu_qwen3_4b_8b_pipeline.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-single_gpu_qwen3_4b_student_qwen3_8b_teacher}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${RUN_NAME}}"
DONE_DIR="${RUN_DIR}/.done"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${DONE_DIR}" "${LOG_DIR}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
SFT_HF_DATASET="${SFT_HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
OPD_HF_DATASET="${OPD_HF_DATASET:-zhuzilin/dapo-math-17k}"
SFT_PROMPTS_SRC="${SFT_PROMPTS_SRC:-}"
OPD_PROMPTS_SRC="${OPD_PROMPTS_SRC:-}"

SFT_SAMPLES="${SFT_SAMPLES:-5000}"
OPD_SAMPLES="${OPD_SAMPLES:-2000}"
SEED="${SEED:-42}"

# Larger curation batch size than the repo default (32). Reduce if vLLM OOMs.
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
CURATION_NUM_GPUS="${CURATION_NUM_GPUS:-1}"
TEACHER_TP="${TEACHER_TP:-1}"
STUDENT_TP="${STUDENT_TP:-1}"
SFT_GEN_MAX_TOKENS="${SFT_GEN_MAX_TOKENS:-4096}"
OPD_ROLLOUT_MAX_TOKENS="${OPD_ROLLOUT_MAX_TOKENS:-4096}"

# Single-GPU SFT settings. Global batch remains 8 by default.
SFT_NUM_NODES="${SFT_NUM_NODES:-1}"
SFT_NUM_GPUS="${SFT_NUM_GPUS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-3000}"
SFT_METRIC_INTERVAL="${SFT_METRIC_INTERVAL:-5}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-1000}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_LR="${SFT_LR:-8e-5}"
SFT_CUTOFF_LEN="${SFT_CUTOFF_LEN:-16384}"
# Chunked training relaunches the trainer with a new max_steps every
# SFT_METRIC_INTERVAL steps. Schedules that depend on max_steps (cosine,
# warmup_ratio) would be recomputed per chunk, so use a chunk-invariant
# schedule: constant LR after a fixed number of warmup steps.
SFT_LR_SCHEDULER="${SFT_LR_SCHEDULER:-constant_with_warmup}"
SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
SFT_ENABLE_LIGER="${SFT_ENABLE_LIGER:-1}"
SFT_REPORT_TO="${SFT_REPORT_TO:-none}"

PROBE_SAMPLES="${PROBE_SAMPLES:-8}"
PROBE_MAX_NEW_TOKENS="${PROBE_MAX_NEW_TOKENS:-512}"
PROBE_TEMPERATURE="${PROBE_TEMPERATURE:-0.7}"
PROBE_TOP_P="${PROBE_TOP_P:-0.9}"

TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.72}"
TEACHER_CONTEXT_LENGTH="${TEACHER_CONTEXT_LENGTH:-8192}"
TEACHER_LOGPROB_CONCURRENCY="${TEACHER_LOGPROB_CONCURRENCY:-16}"
TEACHER_START_TIMEOUT_S="${TEACHER_START_TIMEOUT_S:-900}"

OPD_BATCH_SIZE="${OPD_BATCH_SIZE:-8}"
OPD_NUM_ROLLOUT="${OPD_NUM_ROLLOUT:-250}"
OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU:-8192}"
OPD_SAVE_INTERVAL="${OPD_SAVE_INTERVAL:-25}"
OPD_LR="${OPD_LR:-2e-6}"
OPD_SAVE_DIR="${OPD_SAVE_DIR:-${RUN_DIR}/checkpoints/lightning_opd_single_gpu}"

FORCE="${FORCE:-0}"
SKIP_OPD="${SKIP_OPD:-0}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"

# ── Environment management ──────────────────────────────────────────────────
# USE_CONDA=1: create/activate the README's conda envs per stage.
# USE_CONDA=0: run every stage in the current environment.
USE_CONDA="${USE_CONDA:-1}"
CURATION_ENV="${CURATION_ENV:-curation}"
SFT_ENV="${SFT_ENV:-llamafactory}"
# Empty means: run OPD stages in the current environment (the repo docker
# container ships sglang/Megatron/torch). Set to a conda env name otherwise.
OPD_ENV="${OPD_ENV:-}"
ENV_PYTHON_VERSION="${ENV_PYTHON_VERSION:-3.10}"
# conda-forge avoids the interactive Anaconda default-channel ToS prompt.
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"
# Auto-install Miniconda when conda is missing (USE_CONDA=1 only). Set to 0 to
# error out instead. CONDA_INSTALL_DIR is where a fresh install lands.
AUTO_INSTALL_CONDA="${AUTO_INSTALL_CONDA:-1}"
CONDA_INSTALL_DIR="${CONDA_INSTALL_DIR:-${HOME}/miniconda3}"
# Pin vLLM: the latest builds use a v1 UVA mapped-pinned-host-memory buffer
# (UvaBuffer/get_cuda_view_from_cpu_tensor) that fails on many virtualized
# cloud A100 VMs with "cudaHostGetDevicePointer failed: invalid argument".
# 0.9.2 predates that path and supports Qwen3. Override via VLLM_SPEC.
VLLM_SPEC="${VLLM_SPEC:-vllm==0.9.2}"
CURATION_PACKAGES=("${VLLM_SPEC}" transformers pyarrow pandas tqdm datasets pyyaml "huggingface_hub[cli]")
SFT_PACKAGES=(llamafactory torch transformers datasets pandas pyyaml liger-kernel)

SFT_PROMPTS="${RUN_DIR}/data/prompts/openthoughts3_${SFT_SAMPLES}.jsonl"
OPD_PROMPTS="${RUN_DIR}/data/prompts/opd_prompts_${OPD_SAMPLES}.jsonl"
SFT_RAW_DIR="${RUN_DIR}/data/sft_data_raw"
SFT_PARQUET="${RUN_DIR}/data/sft_data/openthoughts3_${SFT_SAMPLES}_qwen3-8b.parquet"
SFT_CFG_DIR="${RUN_DIR}/configs/sft"
SFT_CONFIG="${SFT_CFG_DIR}/qwen3-4b-single-gpu-runtime.yaml"
SFT_OUTPUT_DIR="${RUN_DIR}/checkpoints/qwen3-4b-base-sft-qwen3-8b"
# Tokenized dataset cache so the 600 chunked trainer relaunches do not
# re-tokenize the SFT data every time. Delete this dir if it gets corrupted
# by a crash mid-tokenization.
SFT_TOKENIZED_DIR="${RUN_DIR}/data/sft_tokenized"
ROLLOUT_RAW_DIR="${RUN_DIR}/data/rollouts_raw"
ROLLOUT_PARQUET="${RUN_DIR}/data/rollouts/dapo_${OPD_SAMPLES}_qwen3-4b-sft-rollouts.parquet"
LIGHTNING_OPD_DIR="${RUN_DIR}/data/lightning_opd"
LIGHTNING_OPD_DATA="${LIGHTNING_OPD_DIR}/$(basename "${ROLLOUT_PARQUET}" .parquet)-lightning-opd-precomputed.parquet"

run_stage() {
  local marker="$1"
  shift
  if [[ "${FORCE}" != "1" && -f "${DONE_DIR}/${marker}" ]]; then
    echo "[skip] ${marker}"
    return 0
  fi
  echo "[stage] ${marker}"
  "$@"
  date -Is > "${DONE_DIR}/${marker}"
}

# ── Environment helpers ─────────────────────────────────────────────────────

CONDA_HOOKED=0

# Detect an already-installed conda that is not yet on PATH (common when a
# previous run installed it but the shell was not reopened).
find_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return 0
  fi
  local candidate
  for candidate in "${CONDA_INSTALL_DIR}" "${HOME}/miniconda3" "${HOME}/anaconda3" \
                   "/opt/conda" "${HOME}/miniforge3"; do
    if [[ -x "${candidate}/bin/conda" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

install_miniconda() {
  local target="${CONDA_INSTALL_DIR}"
  local os arch installer_os installer_arch
  os="$(uname -s)"
  arch="$(uname -m)"
  case "${os}" in
    Linux)  installer_os="Linux" ;;
    Darwin) installer_os="MacOSX" ;;
    *) echo "Unsupported OS for Miniconda auto-install: ${os}" >&2; exit 1 ;;
  esac
  case "${arch}" in
    x86_64|amd64)  installer_arch="x86_64" ;;
    aarch64|arm64) installer_arch="aarch64" ;;
    *) echo "Unsupported arch for Miniconda auto-install: ${arch}" >&2; exit 1 ;;
  esac
  # Apple Silicon uses arm64 in Miniconda filenames, not aarch64.
  [[ "${installer_os}" == "MacOSX" && "${installer_arch}" == "aarch64" ]] && installer_arch="arm64"

  local url="https://repo.anaconda.com/miniconda/Miniconda3-latest-${installer_os}-${installer_arch}.sh"
  local installer="${RUN_DIR}/miniconda_installer.sh"
  echo "[env] conda not found. Installing Miniconda to ${target}"
  echo "[env] Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${installer}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "${url}" -O "${installer}"
  else
    echo "Neither curl nor wget is available to download Miniconda." >&2
    exit 1
  fi
  # -b batch (no prompts), -p prefix. Refuses to clobber an existing dir.
  bash "${installer}" -b -p "${target}"
  rm -f "${installer}"
}

ensure_conda() {
  if (( CONDA_HOOKED )); then
    return 0
  fi

  local base
  if ! base="$(find_conda_base)"; then
    if [[ "${AUTO_INSTALL_CONDA}" == "1" ]]; then
      install_miniconda
      base="${CONDA_INSTALL_DIR}"
    else
      echo "conda not found. Install Miniconda/Anaconda (or set" >&2
      echo "AUTO_INSTALL_CONDA=1), or set USE_CONDA=0 to run every stage in" >&2
      echo "the current environment." >&2
      exit 1
    fi
  fi

  # conda's shell hooks reference unset variables; relax nounset around them.
  set +u
  # shellcheck disable=SC1091
  source "${base}/etc/profile.d/conda.sh"
  set -u
  CONDA_HOOKED=1
}

conda_env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"
}

activate_env() {
  local env_name="$1"
  if [[ "${USE_CONDA}" != "1" || -z "${env_name}" ]]; then
    return 0
  fi
  ensure_conda
  echo "[env] Activating conda env: ${env_name}"
  set +u
  conda activate "${env_name}"
  set -u
}

deactivate_env() {
  local env_name="$1"
  if [[ "${USE_CONDA}" != "1" || -z "${env_name}" ]]; then
    return 0
  fi
  echo "[env] Deactivating conda env: ${env_name}"
  set +u
  conda deactivate
  set -u
}

# Run a command (or shell function) with the given env loaded, then unload it.
# On failure, set -e terminates the script, which drops the activation with it.
# Do not wrap "$@" in a status check here: testing its exit code would disable
# errexit inside the stage function and let partial failures continue.
run_in_env() {
  local env_name="$1"
  shift
  activate_env "${env_name}"
  "$@"
  deactivate_env "${env_name}"
}

# Guarantee pip exists in the given env. Some conda-forge python builds (or
# envs created before pip was requested) ship without pip, which breaks
# `python -m pip`. Safe to call on an already-activated env.
ensure_pip_in_env() {
  local env_name="$1"
  if python -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "[env] pip missing in ${env_name}; bootstrapping via conda"
  conda install -y -n "${env_name}" --override-channels -c "${CONDA_CHANNEL}" pip
}

setup_conda_env() {
  local env_name="$1"
  shift
  if [[ "${USE_CONDA}" != "1" ]]; then
    echo "[env] USE_CONDA=0, skipping setup of ${env_name}"
    return 0
  fi
  ensure_conda
  if ! conda_env_exists "${env_name}"; then
    echo "[env] Creating conda env ${env_name} (python=${ENV_PYTHON_VERSION})"
    conda create -y -n "${env_name}" "python=${ENV_PYTHON_VERSION}" pip \
      --override-channels -c "${CONDA_CHANNEL}"
  fi
  activate_env "${env_name}"
  ensure_pip_in_env "${env_name}"
  echo "[env] Installing into ${env_name}: $*"
  python -m pip install "$@"
  deactivate_env "${env_name}"
}

setup_curation_env() {
  setup_conda_env "${CURATION_ENV}" "${CURATION_PACKAGES[@]}"
}

setup_sft_env() {
  setup_conda_env "${SFT_ENV}" "${SFT_PACKAGES[@]}"
}

setup_opd_env() {
  if [[ "${USE_CONDA}" == "1" && -n "${OPD_ENV}" ]]; then
    ensure_conda
    if ! conda_env_exists "${OPD_ENV}"; then
      echo "[env] Creating conda env ${OPD_ENV} (python=${ENV_PYTHON_VERSION})"
      conda create -y -n "${OPD_ENV}" "python=${ENV_PYTHON_VERSION}" pip \
        --override-channels -c "${CONDA_CHANNEL}"
    fi
    activate_env "${OPD_ENV}"
    ensure_pip_in_env "${OPD_ENV}"
    python -m pip install -e .
    deactivate_env "${OPD_ENV}"
  else
    # Official route: the repo docker container (sglang/Megatron preinstalled);
    # only the repo itself needs installing.
    echo "[env] Installing repo into current environment (pip install -e .)"
    python -m pip install -e .
  fi
}

ensure_prompt_prep_deps() {
  if python - <<'PY'
import importlib.util
import sys

missing = [
    pkg for pkg in ("datasets", "pandas", "pyarrow", "tqdm", "yaml")
    if importlib.util.find_spec(pkg) is None
]
if missing:
    print(" ".join(missing))
    sys.exit(1)
PY
  then
    return 0
  fi

  if [[ "${AUTO_INSTALL_DEPS}" != "1" ]]; then
    echo "Missing prompt-prep Python dependencies. Install them with:" >&2
    echo "  python -m pip install datasets pandas pyarrow tqdm pyyaml" >&2
    exit 1
  fi

  echo "[deps] Installing prompt-prep dependencies: datasets pandas pyarrow tqdm pyyaml"
  python -m pip install datasets pandas pyarrow tqdm pyyaml
}

prepare_sft_prompts() {
  ensure_prompt_prep_deps
  mkdir -p "$(dirname "${SFT_PROMPTS}")"
  if [[ -n "${SFT_PROMPTS_SRC}" ]]; then
    python - "${SFT_PROMPTS_SRC}" "${SFT_PROMPTS}" "${SFT_SAMPLES}" "${SEED}" <<'PY'
import json, random, sys
from pathlib import Path
import pandas as pd

src, dst, count, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = pd.read_parquet(src).to_dict("records") if src.suffix == ".parquet" else [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]

def norm(row):
    if "prompt" in row:
        p = row["prompt"]
        if isinstance(p, str):
            p = [{"role": "user", "content": p}]
        return {"prompt": p}
    if "messages" in row:
        p = [{"role": m["role"], "content": m["content"]} for m in row["messages"] if m["role"] != "assistant"]
        return {"prompt": p} if p else None
    if "conversations" in row:
        p = []
        for t in row["conversations"]:
            role = t.get("from", t.get("role", ""))
            content = t.get("value", t.get("content", ""))
            if role in ("human", "user"):
                p.append({"role": "user", "content": content})
            elif role == "system":
                p.append({"role": "system", "content": content})
        return {"prompt": p} if p else None
    return None

items = [x for x in (norm(r) for r in rows) if x]
if 0 < count < len(items):
    items = random.Random(seed).sample(items, count)
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
  else
    python scripts/prepare_sft_prompts.py \
      --hf-dataset "${SFT_HF_DATASET}" \
      --output "${SFT_PROMPTS}" \
      --num-samples "${SFT_SAMPLES}" \
      --seed "${SEED}"
  fi
}

prepare_opd_prompts() {
  mkdir -p "$(dirname "${OPD_PROMPTS}")"
  local source="${OPD_PROMPTS_SRC}"
  if [[ -z "${source}" ]]; then
    local opd_download="${RUN_DIR}/data/prompts/dapo-math-17k"
    if [[ ! -d "${opd_download}" ]]; then
      if command -v hf >/dev/null 2>&1; then
        hf download --repo-type dataset "${OPD_HF_DATASET}" --include "*.jsonl" --local-dir "${opd_download}"
      else
        huggingface-cli download --repo-type dataset "${OPD_HF_DATASET}" --include "*.jsonl" --local-dir "${opd_download}"
      fi
    fi
    source="$(find "${opd_download}" -name '*.jsonl' | sort | head -n 1)"
  fi

  python - "${source}" "${OPD_PROMPTS}" "${OPD_SAMPLES}" "${SEED}" <<'PY'
import json, random, sys
from pathlib import Path
import pandas as pd

src, dst, count, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = pd.read_parquet(src).to_dict("records") if src.suffix == ".parquet" else [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]

def norm(row):
    p = row.get("prompt") or row.get("messages")
    if isinstance(p, str):
        p = [{"role": "user", "content": p}]
    if p and isinstance(p, list):
        p = [{"role": m["role"], "content": m["content"]} for m in p if m.get("role") != "assistant"]
        return {"prompt": p, "label": str(row.get("label", "0"))}
    return None

items = [x for x in (norm(r) for r in rows) if x]
if 0 < count < len(items):
    items = random.Random(seed).sample(items, count)
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
}

generate_sft_data() {
  mkdir -p "$(dirname "${SFT_PARQUET}")"
  if [[ "${FORCE}" != "1" && -f "${SFT_PARQUET}" ]]; then
    echo "[skip] SFT parquet exists: ${SFT_PARQUET}"
    return 0
  fi
  TEACHER_MODEL="${TEACHER_MODEL}" \
  SFT_PROMPTS="${SFT_PROMPTS}" \
  OUTPUT_DIR="${SFT_RAW_DIR}" \
  NUM_GPUS="${CURATION_NUM_GPUS}" \
  TP_SIZE="${TEACHER_TP}" \
    bash scripts/generate_sft_data.sh \
      --num-samples "${SFT_SAMPLES}" \
      --max-tokens "${SFT_GEN_MAX_TOKENS}" \
      --batch-size "${GEN_BATCH_SIZE}" \
      --checkpoint-dir "${RUN_DIR}/data/.curation_ckpt_sft"

  python data_curation/merge.py --input-dir "${SFT_RAW_DIR}" --output "${SFT_PARQUET}"
}

write_sft_config() {
  mkdir -p "${SFT_CFG_DIR}"
  python - "${SFT_PARQUET}" "${SFT_CFG_DIR}" "${SFT_CONFIG}" <<PY
import json, sys
from pathlib import Path
import yaml

sft_parquet, cfg_dir, config_out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
dataset_name = "openthoughts3_${SFT_SAMPLES}_qwen3_8b_runtime"
(cfg_dir / "dataset_info.json").write_text(json.dumps({
    dataset_name: {
        "file_name": str(sft_parquet),
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
    }
}, indent=2), encoding="utf-8")

with open("configs/sft/qwen3-4b-base-open-thoughts3-qwen3-8b.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# The base YAML references a DeepSpeed config that only exists inside the
# LlamaFactory repo; single-GPU training does not need DeepSpeed at all.
cfg.pop("deepspeed", None)
# warmup_ratio depends on max_steps, which changes per chunk relaunch.
cfg.pop("warmup_ratio", None)

cfg.update({
    "model_name_or_path": "${STUDENT_MODEL}",
    "dataset": dataset_name,
    "max_steps": int("${SFT_MAX_STEPS}"),
    "save_steps": int("${SFT_METRIC_INTERVAL}"),
    "save_total_limit": int("${SFT_SAVE_TOTAL_LIMIT}"),
    "per_device_train_batch_size": int("${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}"),
    "gradient_accumulation_steps": int("${SFT_GRADIENT_ACCUMULATION_STEPS}"),
    "learning_rate": float("${SFT_LR}"),
    "lr_scheduler_type": "${SFT_LR_SCHEDULER}",
    "warmup_steps": int("${SFT_WARMUP_STEPS}"),
    "cutoff_len": int("${SFT_CUTOFF_LEN}"),
    "gradient_checkpointing": True,
    "enable_liger_kernel": "${SFT_ENABLE_LIGER}" == "1",
    "overwrite_cache": False,
    "tokenized_path": "${SFT_TOKENIZED_DIR}",
    "run_name": "${RUN_NAME}-sft",
    "report_to": "${SFT_REPORT_TO}",
})

with config_out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

metric_exists() {
  local step="$1"
  local metrics="${RUN_DIR}/metrics/sft_checkpoint_opd_probe_metrics.jsonl"
  [[ -f "${metrics}" ]] && python - "${metrics}" "${step}" <<'PY'
import json, sys
path, step = sys.argv[1], int(sys.argv[2])
for line in open(path, encoding="utf-8"):
    if line.strip() and int(json.loads(line)["step"]) == step:
        sys.exit(0)
sys.exit(1)
PY
}

latest_checkpoint() {
  find "${SFT_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed -E 's/.*checkpoint-([0-9]+)$/\1 &/' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

run_sft_chunks_with_metrics() {
  mkdir -p "${SFT_OUTPUT_DIR}"
  write_sft_config

  local target
  for (( target=SFT_METRIC_INTERVAL; target<=SFT_MAX_STEPS; target+=SFT_METRIC_INTERVAL )); do
    if metric_exists "${target}"; then
      echo "[skip] SFT metrics already exist for step ${target}"
      local previous_done=$((target - SFT_METRIC_INTERVAL))
      if (( previous_done > 0 )) && [[ -d "${SFT_OUTPUT_DIR}/checkpoint-${target}" ]]; then
        rm -rf "${SFT_OUTPUT_DIR}/checkpoint-${previous_done}"
      fi
      continue
    fi

    local ckpt="${SFT_OUTPUT_DIR}/checkpoint-${target}"
    if [[ ! -d "${ckpt}" ]]; then
      local resume_ckpt
      resume_ckpt="$(latest_checkpoint || true)"
      echo "[sft] Training/resuming to step ${target}"

      local extra_args=(
        "max_steps=${target}"
        "save_steps=${SFT_METRIC_INTERVAL}"
        "save_total_limit=${SFT_SAVE_TOTAL_LIMIT}"
      )
      if [[ -n "${resume_ckpt}" ]]; then
        extra_args+=("resume_from_checkpoint=${resume_ckpt}")
      else
        extra_args+=("overwrite_output_dir=true")
      fi

      torchrun \
        --nnodes "${SFT_NUM_NODES}" \
        --nproc_per_node "${SFT_NUM_GPUS}" \
        --rdzv_id "${RANDOM}" \
        --rdzv_backend c10d \
        --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        -m llamafactory.cli.train \
        "${SFT_CONFIG}" \
        "dataset_dir=${SFT_CFG_DIR}" \
        "output_dir=${SFT_OUTPUT_DIR}" \
        "${extra_args[@]}" \
        2>&1 | tee "${LOG_DIR}/sft_to_step_${target}.log"
    fi

    python scripts/run_official_qwen3_4b_8b_with_sft_metrics.py \
      --metrics-only "${SFT_OUTPUT_DIR}" \
      --run-name "${RUN_NAME}" \
      --run-dir "${RUN_DIR}" \
      --opd-prompts "${OPD_PROMPTS}" \
      --opd-samples "${OPD_SAMPLES}" \
      --probe-samples "${PROBE_SAMPLES}" \
      --probe-max-new-tokens "${PROBE_MAX_NEW_TOKENS}" \
      --probe-temperature "${PROBE_TEMPERATURE}" \
      --probe-top-p "${PROBE_TOP_P}" \
      --teacher-model "${TEACHER_MODEL}"

    local previous=$((target - SFT_METRIC_INTERVAL))
    if (( previous > 0 )); then
      rm -rf "${SFT_OUTPUT_DIR}/checkpoint-${previous}"
    fi
  done

  echo "${SFT_OUTPUT_DIR}/checkpoint-${SFT_MAX_STEPS}" > "${RUN_DIR}/SFT_CHECKPOINT.txt"
}

collect_rollouts() {
  local sft_ckpt
  sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  mkdir -p "$(dirname "${ROLLOUT_PARQUET}")"
  if [[ "${FORCE}" != "1" && -f "${ROLLOUT_PARQUET}" ]]; then
    echo "[skip] rollout parquet exists: ${ROLLOUT_PARQUET}"
    return 0
  fi

  SFT_CHECKPOINT="${sft_ckpt}" \
  OPD_PROMPTS="${OPD_PROMPTS}" \
  OUTPUT_DIR="${ROLLOUT_RAW_DIR}" \
  NUM_GPUS="${CURATION_NUM_GPUS}" \
  TP_SIZE="${STUDENT_TP}" \
    bash scripts/collect_rollouts.sh \
      --num-samples "${OPD_SAMPLES}" \
      --max-tokens "${OPD_ROLLOUT_MAX_TOKENS}" \
      --batch-size "${ROLLOUT_BATCH_SIZE}" \
      --checkpoint-dir "${RUN_DIR}/data/.curation_ckpt_rollout"

  python data_curation/merge.py --input-dir "${ROLLOUT_RAW_DIR}" --output "${ROLLOUT_PARQUET}"
}

start_teacher_server() {
  local teacher_log="${LOG_DIR}/sglang_teacher_8b.log"
  python -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" \
    --host 127.0.0.1 \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP}" \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
    --context-length "${TEACHER_CONTEXT_LENGTH}" \
    > "${teacher_log}" 2>&1 &
  TEACHER_PID="$!"

  local waited=0
  until curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; do
    if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
      echo "Teacher server exited early. Log: ${teacher_log}" >&2
      exit 1
    fi
    if (( waited >= TEACHER_START_TIMEOUT_S )); then
      echo "Teacher server timeout. Log: ${teacher_log}" >&2
      exit 1
    fi
    echo "Waiting for teacher server..."
    tail -n 5 "${teacher_log}" || true
    sleep 5
    waited=$((waited + 5))
  done
}

precompute_lightning_opd() {
  mkdir -p "${LIGHTNING_OPD_DIR}"
  if [[ "${FORCE}" != "1" && -f "${LIGHTNING_OPD_DATA}" ]]; then
    echo "[skip] Lightning OPD data exists: ${LIGHTNING_OPD_DATA}"
    return 0
  fi

  local sft_ckpt
  sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  local TEACHER_PID=""
  start_teacher_server

  set +e
  python data_curation/prepare_lightning_opd.py \
    --tokenizer-path "${sft_ckpt}" \
    --input-parquet "${ROLLOUT_PARQUET}" \
    --output-dir "${LIGHTNING_OPD_DIR}" \
    --max-response-len "${OPD_ROLLOUT_MAX_TOKENS}" \
    --compute-teacher-logprobs \
    --teacher-url "http://127.0.0.1:${TEACHER_PORT}/generate" \
    --concurrency "${TEACHER_LOGPROB_CONCURRENCY}"
  local status=$?
  [[ -n "${TEACHER_PID:-}" ]] && kill "${TEACHER_PID}" 2>/dev/null || true
  wait "${TEACHER_PID}" 2>/dev/null || true
  set -e
  return "${status}"
}

run_lightning_opd_single_gpu() {
  local sft_ckpt
  sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  SFT_CHECKPOINT="${sft_ckpt}" \
  LIGHTNING_OPD_DATA="${LIGHTNING_OPD_DATA}" \
  LIGHTNING_OPD_SAVE_DIR="${OPD_SAVE_DIR}" \
  LIGHTNING_OPD_NUM_ROLLOUT="${OPD_NUM_ROLLOUT}" \
  LIGHTNING_OPD_BATCH_SIZE="${OPD_BATCH_SIZE}" \
  LIGHTNING_OPD_MAX_RESPONSE_LEN="${OPD_ROLLOUT_MAX_TOKENS}" \
  LIGHTNING_OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU}" \
  LIGHTNING_OPD_SAVE_INTERVAL="${OPD_SAVE_INTERVAL}" \
  LIGHTNING_OPD_LR="${OPD_LR}" \
    python configs/lightning_opd/qwen3-4b-lightning-opd-single-gpu.py
}

run_stage "setup_env_curation" setup_curation_env
run_stage "setup_env_sft" setup_sft_env

run_stage "prepare_sft_prompts" run_in_env "${CURATION_ENV}" prepare_sft_prompts
run_stage "prepare_opd_prompts" run_in_env "${CURATION_ENV}" prepare_opd_prompts
run_stage "generate_sft_data" run_in_env "${CURATION_ENV}" generate_sft_data
run_stage "sft_chunks_and_metrics" run_in_env "${SFT_ENV}" run_sft_chunks_with_metrics

if [[ "${SKIP_OPD}" == "1" ]]; then
  echo "[done] SFT checkpoint: $(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  echo "[done] SFT metrics: ${RUN_DIR}/metrics/sft_checkpoint_opd_probe_metrics.csv"
  exit 0
fi

run_stage "collect_rollouts" run_in_env "${CURATION_ENV}" collect_rollouts
run_stage "setup_env_opd" setup_opd_env
run_stage "precompute_lightning_opd" run_in_env "${OPD_ENV}" precompute_lightning_opd
run_stage "lightning_opd_train_single_gpu" run_in_env "${OPD_ENV}" run_lightning_opd_single_gpu

echo "[done] SFT checkpoint: $(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
echo "[done] SFT metrics: ${RUN_DIR}/metrics/sft_checkpoint_opd_probe_metrics.csv"
echo "[done] Lightning OPD data: ${LIGHTNING_OPD_DATA}"
