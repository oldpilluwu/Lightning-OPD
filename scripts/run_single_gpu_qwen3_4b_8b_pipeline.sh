#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end Lightning-OPD pipeline on a SINGLE GPU.
# Student: Qwen3-4B-Base   Teacher: Qwen3-8B   SFT: 5k   OPD: 2k
#
# This is just the README's Step 0 -> 6, wired into one resumable script:
#   0. prepare SFT prompts (OpenThoughts3) + OPD prompts (DAPO-Math-17k)
#   1. generate SFT data with the teacher (vLLM, large batch)
#   2. SFT the student in 5-step chunks; after each chunk score the checkpoint
#      on a fixed teacher OPD probe (teacher NLL, student NLL, KL, policy
#      drift) and delete the previous checkpoint
#   3. collect student rollouts on the OPD prompts
#   4. precompute teacher log-probs (starts/stops an sglang teacher server)
#   5. Lightning OPD training
#   6. convert the Megatron checkpoint to HuggingFace format
#
# Resumable: every stage records a marker under RUN_DIR/.done and is skipped
# on rerun; SFT resumes from the last kept checkpoint; already-computed metric
# rows are skipped. Just rerun the same command after any interruption.
#
# Optional per-stage conda envs (leave unset to use the current env):
#   CURATION_ENV  used for steps 0,1,3   (vLLM)
#   SFT_ENV       used for step 2         (LlamaFactory)
#   OPD_ENV       used for steps 4,5,6    (sglang + Megatron / slime)
#
# Usage:
#   bash scripts/run_single_gpu_qwen3_4b_8b_pipeline.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ── Config (override any of these via environment) ──────────────────────────
RUN_NAME="${RUN_NAME:-single_gpu_qwen3_4b_8b}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${RUN_NAME}}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
SFT_HF_DATASET="${SFT_HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
OPD_HF_DATASET="${OPD_HF_DATASET:-zhuzilin/dapo-math-17k}"

SFT_SAMPLES="${SFT_SAMPLES:-5000}"
OPD_SAMPLES="${OPD_SAMPLES:-2000}"
SEED="${SEED:-42}"

# Prompt generation: large batch as requested. Reduce if vLLM OOMs.
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-4096}"

# SFT (single GPU). Metrics every SFT_METRIC_INTERVAL steps.
SFT_MAX_STEPS="${SFT_MAX_STEPS:-500}"
SFT_METRIC_INTERVAL="${SFT_METRIC_INTERVAL:-5}"
MASTER_PORT="${MASTER_PORT:-29500}"

# OPD probe used for the per-checkpoint metrics.
PROBE_SAMPLES="${PROBE_SAMPLES:-16}"
PROBE_MAX_NEW_TOKENS="${PROBE_MAX_NEW_TOKENS:-512}"
PROBE_TEMPERATURE="${PROBE_TEMPERATURE:-0.7}"
PROBE_TOP_P="${PROBE_TOP_P:-0.9}"

# Teacher server (step 4).
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.72}"
TEACHER_CONTEXT_LENGTH="${TEACHER_CONTEXT_LENGTH:-8192}"
TEACHER_START_TIMEOUT_S="${TEACHER_START_TIMEOUT_S:-900}"
LOGPROB_CONCURRENCY="${LOGPROB_CONCURRENCY:-16}"

# Lightning OPD (step 5).
OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN:-4096}"
OPD_BATCH_SIZE="${OPD_BATCH_SIZE:-8}"
OPD_NUM_ROLLOUT="${OPD_NUM_ROLLOUT:-250}"
OPD_SAVE_INTERVAL="${OPD_SAVE_INTERVAL:-25}"
OPD_LR="${OPD_LR:-2e-6}"

# Optional conda envs (empty => current env).
CURATION_ENV="${CURATION_ENV:-}"
SFT_ENV="${SFT_ENV:-}"
OPD_ENV="${OPD_ENV:-}"

SKIP_OPD="${SKIP_OPD:-0}"    # 1 => stop after SFT + metrics
SKIP_CONVERT="${SKIP_CONVERT:-0}"

# ── Derived paths ───────────────────────────────────────────────────────────
DONE_DIR="${RUN_DIR}/.done"
LOG_DIR="${RUN_DIR}/logs"
METRICS_DIR="${RUN_DIR}/metrics"
PROMPT_DIR="${RUN_DIR}/data/prompts"
mkdir -p "${DONE_DIR}" "${LOG_DIR}" "${METRICS_DIR}" "${PROMPT_DIR}"

SFT_PROMPTS="${PROMPT_DIR}/openthoughts3_${SFT_SAMPLES}.jsonl"
OPD_PROMPTS="${PROMPT_DIR}/opd_${OPD_SAMPLES}.jsonl"

# Exported for the embedded metrics scorer (score_checkpoint reads os.environ).
export TEACHER_MODEL OPD_PROMPTS METRICS_DIR SEED
export PROBE_SAMPLES PROBE_MAX_NEW_TOKENS PROBE_TEMPERATURE PROBE_TOP_P
OPD_DOWNLOAD_DIR="${PROMPT_DIR}/dapo-math-17k"

SFT_RAW_DIR="${RUN_DIR}/data/sft_data_raw"
SFT_PARQUET="${RUN_DIR}/data/sft_data/openthoughts3_${SFT_SAMPLES}_qwen3-8b.parquet"
SFT_CFG_DIR="${RUN_DIR}/configs/sft"
SFT_CONFIG="${SFT_CFG_DIR}/qwen3-4b-single-gpu-runtime.yaml"
SFT_OUTPUT_DIR="${RUN_DIR}/checkpoints/qwen3-4b-base-sft-qwen3-8b"
SFT_TOKENIZED_DIR="${RUN_DIR}/data/sft_tokenized"

ROLLOUT_RAW_DIR="${RUN_DIR}/data/rollouts_raw"
ROLLOUT_PARQUET="${RUN_DIR}/data/rollouts/dapo_${OPD_SAMPLES}_qwen3-4b-sft-rollouts.parquet"
LIGHTNING_OPD_DIR="${RUN_DIR}/data/lightning_opd"
LIGHTNING_OPD_DATA="${LIGHTNING_OPD_DIR}/$(basename "${ROLLOUT_PARQUET}" .parquet)-lightning-opd-precomputed.parquet"
OPD_SAVE_DIR="${RUN_DIR}/checkpoints/lightning_opd"
HF_OUTPUT_DIR="${RUN_DIR}/checkpoints/qwen3-4b-lightning-opd-hf"

# ── Helpers ─────────────────────────────────────────────────────────────────
run_stage() {  # run_stage <marker> <cmd...>
  local marker="$1"; shift
  if [[ -f "${DONE_DIR}/${marker}" ]]; then
    echo "[skip] ${marker}"
    return 0
  fi
  echo "[stage] ${marker}"
  "$@"
  date -Is > "${DONE_DIR}/${marker}"
}

# Activate a conda env for the duration of a command, if one is configured.
in_env() {  # in_env <env-name> <cmd...>
  local env_name="$1"; shift
  if [[ -z "${env_name}" ]]; then
    "$@"
    return
  fi
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${env_name}"
  set -u
  "$@"
  set +u
  conda deactivate
  set -u
}

# ── Step 0: prompts ─────────────────────────────────────────────────────────
prepare_sft_prompts() {
  python scripts/prepare_sft_prompts.py \
    --hf-dataset "${SFT_HF_DATASET}" \
    --output "${SFT_PROMPTS}" \
    --num-samples "${SFT_SAMPLES}" \
    --seed "${SEED}"
}

prepare_opd_prompts() {
  if [[ ! -d "${OPD_DOWNLOAD_DIR}" ]]; then
    if command -v hf >/dev/null 2>&1; then
      hf download --repo-type dataset "${OPD_HF_DATASET}" --include "*.jsonl" --local-dir "${OPD_DOWNLOAD_DIR}"
    else
      huggingface-cli download --repo-type dataset "${OPD_HF_DATASET}" --include "*.jsonl" --local-dir "${OPD_DOWNLOAD_DIR}"
    fi
  fi
  local src
  src="$(find "${OPD_DOWNLOAD_DIR}" -name '*.jsonl' | sort | head -n 1)"
  python - "${src}" "${OPD_PROMPTS}" "${OPD_SAMPLES}" "${SEED}" <<'PY'
import json, random, sys
from pathlib import Path
src, dst, count, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
def norm(r):
    p = r.get("prompt") or r.get("messages")
    if isinstance(p, str):
        p = [{"role": "user", "content": p}]
    if isinstance(p, list) and p:
        p = [{"role": m["role"], "content": m["content"]} for m in p if m.get("role") != "assistant"]
        return {"prompt": p, "label": str(r.get("label", "0"))}
    return None
items = [x for x in (norm(r) for r in rows) if x]
if 0 < count < len(items):
    items = random.Random(seed).sample(items, count)
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"Wrote {len(items)} OPD prompts -> {dst}")
PY
}

# ── Step 1: SFT data generation (teacher) ───────────────────────────────────
generate_sft_data() {
  mkdir -p "$(dirname "${SFT_PARQUET}")"
  TEACHER_MODEL="${TEACHER_MODEL}" \
  SFT_PROMPTS="${SFT_PROMPTS}" \
  OUTPUT_DIR="${SFT_RAW_DIR}" \
  NUM_GPUS=1 TP_SIZE=1 \
    bash scripts/generate_sft_data.sh \
      --num-samples "${SFT_SAMPLES}" \
      --max-tokens "${GEN_MAX_TOKENS}" \
      --batch-size "${GEN_BATCH_SIZE}" \
      --checkpoint-dir "${RUN_DIR}/data/.curation_ckpt_sft"
  python data_curation/merge.py --input-dir "${SFT_RAW_DIR}" --output "${SFT_PARQUET}"
}

# ── Step 2: SFT + per-checkpoint metrics ────────────────────────────────────
write_sft_config() {
  mkdir -p "${SFT_CFG_DIR}"
  python - "${SFT_PARQUET}" "${SFT_CFG_DIR}" "${SFT_CONFIG}" \
           "${STUDENT_MODEL}" "${SFT_SAMPLES}" "${SFT_METRIC_INTERVAL}" \
           "${SFT_TOKENIZED_DIR}" "${RUN_NAME}" <<'PY'
import json, sys
from pathlib import Path
import yaml
parquet, cfg_dir, cfg_out, model, samples, interval, tok_dir, run_name = sys.argv[1:9]
cfg_dir = Path(cfg_dir)
dataset_name = f"sft_{samples}_qwen3_8b"
(cfg_dir / "dataset_info.json").write_text(json.dumps({
    dataset_name: {"file_name": str(Path(parquet).resolve()),
                   "formatting": "sharegpt",
                   "columns": {"messages": "messages"},
                   # our messages use role/content keys, not sharegpt's from/value
                   "tags": {"role_tag": "role",
                            "content_tag": "content",
                            "user_tag": "user",
                            "assistant_tag": "assistant",
                            "system_tag": "system"}}
}, indent=2), encoding="utf-8")

with open("configs/sft/qwen3-4b-base-open-thoughts3-qwen3-8b.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
# deepspeed config path only exists inside the LlamaFactory repo; not needed on 1 GPU.
cfg.pop("deepspeed", None)
# warmup_ratio/cosine depend on max_steps, which changes every chunk; use a
# chunk-invariant constant schedule instead.
cfg.pop("warmup_ratio", None)
cfg.update({
    "model_name_or_path": model,
    "dataset": dataset_name,
    "save_steps": int(interval),
    "save_total_limit": 1000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "gradient_checkpointing": True,
    "overwrite_cache": False,
    "tokenized_path": tok_dir,
    "run_name": f"{run_name}-sft",
    "report_to": "none",
})
with open(cfg_out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

latest_checkpoint() {
  find "${SFT_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed -E 's/.*checkpoint-([0-9]+)$/\1 &/' | sort -n | tail -n 1 | cut -d' ' -f2-
}

metric_exists() {  # metric_exists <step>
  local f="${METRICS_DIR}/sft_opd_probe_metrics.jsonl"
  [[ -f "${f}" ]] && python - "${f}" "$1" <<'PY'
import json, sys
path, step = sys.argv[1], int(sys.argv[2])
for line in open(path, encoding="utf-8"):
    if line.strip() and int(json.loads(line)["step"]) == step:
        sys.exit(0)
sys.exit(1)
PY
}

run_sft_with_metrics() {
  mkdir -p "${SFT_OUTPUT_DIR}"
  write_sft_config
  local target
  for (( target=SFT_METRIC_INTERVAL; target<=SFT_MAX_STEPS; target+=SFT_METRIC_INTERVAL )); do
    local previous=$((target - SFT_METRIC_INTERVAL))
    if metric_exists "${target}"; then
      echo "[skip] SFT metrics step ${target}"
      [[ ${previous} -gt 0 ]] && rm -rf "${SFT_OUTPUT_DIR}/checkpoint-${previous}"
      continue
    fi

    if [[ ! -d "${SFT_OUTPUT_DIR}/checkpoint-${target}" ]]; then
      local resume; resume="$(latest_checkpoint || true)"
      local resume_arg
      if [[ -n "${resume}" ]]; then
        resume_arg="resume_from_checkpoint=${resume}"
      else
        resume_arg="overwrite_output_dir=true"
      fi
      echo "[sft] training to step ${target} (resume: ${resume:-none})"
      torchrun --nnodes 1 --nproc_per_node 1 \
        --rdzv_id "${RANDOM}" --rdzv_backend c10d --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
        -m llamafactory.launcher "${SFT_CONFIG}" \
        "dataset_dir=${SFT_CFG_DIR}" \
        "output_dir=${SFT_OUTPUT_DIR}" \
        "max_steps=${target}" \
        "${resume_arg}" \
        2>&1 | tee "${LOG_DIR}/sft_step_${target}.log"
    fi

    score_checkpoint "${SFT_OUTPUT_DIR}/checkpoint-${target}" "${target}"
    [[ ${previous} -gt 0 ]] && rm -rf "${SFT_OUTPUT_DIR}/checkpoint-${previous}"
  done
  echo "${SFT_OUTPUT_DIR}/checkpoint-${SFT_MAX_STEPS}" > "${RUN_DIR}/SFT_CHECKPOINT.txt"
}

# Score one checkpoint on the fixed teacher OPD probe and append a metrics row.
score_checkpoint() {  # score_checkpoint <checkpoint-dir> <step>
  python - "$1" "$2" <<'PY'
import csv, gc, json, os, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt, step = sys.argv[1], int(sys.argv[2])
teacher_model = os.environ["TEACHER_MODEL"]
opd_prompts   = Path(os.environ["OPD_PROMPTS"])
metrics_dir   = Path(os.environ["METRICS_DIR"])
n_probe       = int(os.environ["PROBE_SAMPLES"])
max_new       = int(os.environ["PROBE_MAX_NEW_TOKENS"])
temperature   = float(os.environ["PROBE_TEMPERATURE"])
top_p         = float(os.environ["PROBE_TOP_P"])
seed          = int(os.environ["SEED"])

probe_path = metrics_dir / "opd_teacher_probe.jsonl"
prev_path  = metrics_dir / "prev_student_logps.json"
jsonl_path = metrics_dir / "sft_opd_probe_metrics.jsonl"
csv_path   = metrics_dir / "sft_opd_probe_metrics.csv"

def token_logps(model, ids, prompt_len):
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, :-1].float()
        targets = x[0, 1:]
        start = max(prompt_len - 1, 0)
        n = len(ids) - prompt_len
        lp = torch.log_softmax(logits[start:start + n], dim=-1)
        return lp.gather(-1, targets[start:start + n].unsqueeze(-1)).squeeze(-1).cpu().tolist()

def load(model_id):
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
            device_map={"": "cuda"}, trust_remote_code=True, attn_implementation="flash_attention_2")
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
            device_map={"": "cuda"}, trust_remote_code=True, attn_implementation="sdpa")

tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# Build the fixed probe once (teacher generates responses to OPD prompts).
if not probe_path.exists():
    random.seed(seed); torch.manual_seed(seed)
    rows = [json.loads(x) for x in opd_prompts.read_text(encoding="utf-8").splitlines() if x.strip()][:n_probe]
    teacher = load(teacher_model); teacher.eval()
    probe = []
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pids = tok.encode(text, add_special_tokens=False)
        x = torch.tensor([pids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = teacher.generate(input_ids=x, max_new_tokens=max_new, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)[0].tolist()
        rids = out[len(pids):]
        if not rids:
            continue
        tlp = token_logps(teacher, pids + rids, len(pids))
        probe.append({"prompt_ids": pids, "response_ids": rids, "teacher_logps": tlp})
    probe_path.write_text("\n".join(json.dumps(p) for p in probe), encoding="utf-8")
    del teacher; gc.collect(); torch.cuda.empty_cache()

probe = [json.loads(x) for x in probe_path.read_text(encoding="utf-8").splitlines() if x.strip()]

# Score the student checkpoint.
model = load(ckpt); model.eval()
t_nll = s_nll = 0.0; ntok = 0
cur = []
for it in probe:
    ids = it["prompt_ids"] + it["response_ids"]
    lp = token_logps(model, ids, len(it["prompt_ids"]))
    cur.append(lp)
    t_nll += -sum(it["teacher_logps"]); s_nll += -sum(lp); ntok += len(lp)
del model; gc.collect(); torch.cuda.empty_cache()

# Inter-step policy drift vs the previously scored checkpoint.
drift_signed = drift_abs = 0.0; dtok = 0
if prev_path.exists():
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    for pv, cv in zip(prev, cur):
        for a, b in zip(pv, cv):
            drift_signed += a - b; drift_abs += abs(a - b); dtok += 1
prev_path.write_text(json.dumps(cur), encoding="utf-8")

ntok = max(ntok, 1); dtok = max(dtok, 1)
teacher_nll = t_nll / ntok; student_nll = s_nll / ntok
row = {
    "step": step,
    "tokens": ntok,
    "teacher_nll": teacher_nll,
    "student_nll": student_nll,
    "kl_teacher_to_student": student_nll - teacher_nll,
    "policy_drift_signed": drift_signed / dtok,
    "policy_drift_abs": drift_abs / dtok,
}
with jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
write_header = not csv_path.exists()
with csv_path.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print("[metrics] " + json.dumps(row, sort_keys=True))
PY
}

# ── Step 3: student rollouts ────────────────────────────────────────────────
collect_rollouts() {
  local sft_ckpt; sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  mkdir -p "$(dirname "${ROLLOUT_PARQUET}")"
  SFT_CHECKPOINT="${sft_ckpt}" \
  OPD_PROMPTS="${OPD_PROMPTS}" \
  OUTPUT_DIR="${ROLLOUT_RAW_DIR}" \
  NUM_GPUS=1 TP_SIZE=1 \
    bash scripts/collect_rollouts.sh \
      --num-samples "${OPD_SAMPLES}" \
      --max-tokens "${OPD_MAX_RESPONSE_LEN}" \
      --batch-size "${GEN_BATCH_SIZE}" \
      --checkpoint-dir "${RUN_DIR}/data/.curation_ckpt_rollout"
  python data_curation/merge.py --input-dir "${ROLLOUT_RAW_DIR}" --output "${ROLLOUT_PARQUET}"
}

# ── Step 4: precompute teacher logprobs ─────────────────────────────────────
precompute_teacher_logprobs() {
  local sft_ckpt; sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  mkdir -p "${LIGHTNING_OPD_DIR}"

  # Phase 1: tokenize (CPU).
  python data_curation/prepare_lightning_opd.py \
    --tokenizer-path "${sft_ckpt}" \
    --input-parquet "${ROLLOUT_PARQUET}" \
    --output-dir "${LIGHTNING_OPD_DIR}" \
    --max-response-len "${OPD_MAX_RESPONSE_LEN}"

  # Phase 2: teacher server + logprobs.
  local teacher_log="${LOG_DIR}/sglang_teacher_8b.log"
  python -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" \
    --host 127.0.0.1 --port "${TEACHER_PORT}" --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
    --context-length "${TEACHER_CONTEXT_LENGTH}" \
    > "${teacher_log}" 2>&1 &
  local teacher_pid="$!"
  trap 'kill "${teacher_pid}" 2>/dev/null || true' RETURN

  local waited=0
  until curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; do
    if ! kill -0 "${teacher_pid}" 2>/dev/null; then
      echo "Teacher server exited early. Log: ${teacher_log}" >&2; exit 1
    fi
    if (( waited >= TEACHER_START_TIMEOUT_S )); then
      echo "Teacher server timeout. Log: ${teacher_log}" >&2; exit 1
    fi
    echo "Waiting for teacher server..."; sleep 5; waited=$((waited + 5))
  done

  python data_curation/prepare_lightning_opd.py \
    --tokenizer-path "${sft_ckpt}" \
    --input-parquet "${ROLLOUT_PARQUET}" \
    --output-dir "${LIGHTNING_OPD_DIR}" \
    --max-response-len "${OPD_MAX_RESPONSE_LEN}" \
    --compute-teacher-logprobs \
    --teacher-url "http://127.0.0.1:${TEACHER_PORT}/generate" \
    --concurrency "${LOGPROB_CONCURRENCY}"
}

# ── Step 5: Lightning OPD training ──────────────────────────────────────────
lightning_opd_train() {
  local sft_ckpt; sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  SFT_CHECKPOINT="${sft_ckpt}" \
  LIGHTNING_OPD_DATA="${LIGHTNING_OPD_DATA}" \
  LIGHTNING_OPD_SAVE_DIR="${OPD_SAVE_DIR}" \
  LIGHTNING_OPD_NUM_ROLLOUT="${OPD_NUM_ROLLOUT}" \
  LIGHTNING_OPD_BATCH_SIZE="${OPD_BATCH_SIZE}" \
  LIGHTNING_OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN}" \
  LIGHTNING_OPD_SAVE_INTERVAL="${OPD_SAVE_INTERVAL}" \
  LIGHTNING_OPD_LR="${OPD_LR}" \
    python configs/lightning_opd/qwen3-4b-lightning-opd-single-gpu.py
}

# ── Step 6: convert Megatron -> HF ──────────────────────────────────────────
convert_to_hf() {
  local sft_ckpt; sft_ckpt="$(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  local iter_dir
  iter_dir="$(find "${OPD_SAVE_DIR}" -maxdepth 1 -type d -name 'iter_*' 2>/dev/null | sort | tail -n 1)"
  if [[ -z "${iter_dir}" ]]; then
    echo "No iter_* checkpoint found under ${OPD_SAVE_DIR}; skipping conversion." >&2
    return 0
  fi
  MEGATRON_CKPT_DIR="${iter_dir}" \
  HF_OUTPUT_DIR="${HF_OUTPUT_DIR}" \
  ORIGIN_HF_DIR="${sft_ckpt}" \
    bash scripts/convert_megatron_to_hf.sh
}

# ── Pipeline ────────────────────────────────────────────────────────────────
run_stage "step0_sft_prompts"   in_env "${CURATION_ENV}" prepare_sft_prompts
run_stage "step0_opd_prompts"   in_env "${CURATION_ENV}" prepare_opd_prompts
run_stage "step1_generate_sft"  in_env "${CURATION_ENV}" generate_sft_data
run_stage "step2_sft_metrics"   in_env "${SFT_ENV}"      run_sft_with_metrics

if [[ "${SKIP_OPD}" == "1" ]]; then
  echo "[done] SFT checkpoint: $(cat "${RUN_DIR}/SFT_CHECKPOINT.txt")"
  echo "[done] SFT metrics:    ${METRICS_DIR}/sft_opd_probe_metrics.csv"
  exit 0
fi

run_stage "step3_rollouts"      in_env "${CURATION_ENV}" collect_rollouts
run_stage "step4_teacher_logp"  in_env "${OPD_ENV}"      precompute_teacher_logprobs
run_stage "step5_lightning_opd" in_env "${OPD_ENV}"      lightning_opd_train
[[ "${SKIP_CONVERT}" == "1" ]] || run_stage "step6_convert_hf" in_env "${OPD_ENV}" convert_to_hf

echo "[done] SFT metrics:        ${METRICS_DIR}/sft_opd_probe_metrics.csv"
echo "[done] Lightning OPD data: ${LIGHTNING_OPD_DATA}"
echo "[done] OPD checkpoints:    ${OPD_SAVE_DIR}"
echo "[done] HF model:           ${HF_OUTPUT_DIR}"
