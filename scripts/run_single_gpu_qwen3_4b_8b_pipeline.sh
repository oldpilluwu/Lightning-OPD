#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Single-GPU Qwen3-4B student / Qwen3-8B teacher Lightning-OPD pipeline.
#
# Resumability:
#   - Every stage writes under RUN_DIR and skips completed outputs.
#   - SFT is run in 5-step chunks by default. After each checkpoint is saved,
#     the checkpoint is scored, then the previous checkpoint is removed.
#   - The latest checkpoint is retained because it is required for resume.
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
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
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
SFT_REPORT_TO="${SFT_REPORT_TO:-wandb}"

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

SFT_PROMPTS="${RUN_DIR}/data/prompts/openthoughts3_${SFT_SAMPLES}.jsonl"
OPD_PROMPTS="${RUN_DIR}/data/prompts/opd_prompts_${OPD_SAMPLES}.jsonl"
SFT_RAW_DIR="${RUN_DIR}/data/sft_data_raw"
SFT_PARQUET="${RUN_DIR}/data/sft_data/openthoughts3_${SFT_SAMPLES}_qwen3-8b.parquet"
SFT_CFG_DIR="${RUN_DIR}/configs/sft"
SFT_CONFIG="${SFT_CFG_DIR}/qwen3-4b-single-gpu-runtime.yaml"
SFT_OUTPUT_DIR="${RUN_DIR}/checkpoints/qwen3-4b-base-sft-qwen3-8b"
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
      --batch-size "${GEN_BATCH_SIZE}"

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

cfg.update({
    "model_name_or_path": "${STUDENT_MODEL}",
    "dataset": dataset_name,
    "max_steps": int("${SFT_MAX_STEPS}"),
    "save_steps": int("${SFT_METRIC_INTERVAL}"),
    "save_total_limit": int("${SFT_SAVE_TOTAL_LIMIT}"),
    "per_device_train_batch_size": int("${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}"),
    "gradient_accumulation_steps": int("${SFT_GRADIENT_ACCUMULATION_STEPS}"),
    "learning_rate": float("${SFT_LR}"),
    "cutoff_len": int("${SFT_CUTOFF_LEN}"),
    "gradient_checkpointing": True,
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
      --batch-size "${ROLLOUT_BATCH_SIZE}"

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

run_stage "prepare_sft_prompts" prepare_sft_prompts
run_stage "prepare_opd_prompts" prepare_opd_prompts
run_stage "generate_sft_data" generate_sft_data
run_stage "sft_chunks_and_metrics" run_sft_chunks_with_metrics

if [[ "${SKIP_OPD}" == "1" ]]; then
  echo "[done] SFT checkpoint: $(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  echo "[done] SFT metrics: ${RUN_DIR}/metrics/sft_checkpoint_opd_probe_metrics.csv"
  exit 0
fi

run_stage "collect_rollouts" collect_rollouts
run_stage "precompute_lightning_opd" precompute_lightning_opd
run_stage "lightning_opd_train_single_gpu" run_lightning_opd_single_gpu

echo "[done] SFT checkpoint: $(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
echo "[done] SFT metrics: ${RUN_DIR}/metrics/sft_checkpoint_opd_probe_metrics.csv"
echo "[done] Lightning OPD data: ${LIGHTNING_OPD_DATA}"
