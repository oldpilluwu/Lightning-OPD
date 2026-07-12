#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Lightning OPD on AWS Trainium — single resumable entrypoint.
#
# Runs the full pipeline (steps 0-6) for one scale on a single Trainium
# instance. Defaults target trn2.3xlarge (1 Trainium2 chip); override
# NUM_CORES / TRAIN_TP / GEN_TP for a bigger box (see the profile block
# below). Every stage writes a marker into data/.pipeline_state/ when it
# completes; re-running the script resumes at the first unfinished stage
# (and the generation / scoring stages also resume mid-stage from their own
# checkpoints).
#
# Usage:
#   SCALE=4b bash trainium/run_pipeline.sh          # Qwen3-4B-Base + Qwen3-8B teacher
#   SCALE=4b SMOKE=1 bash trainium/run_pipeline.sh  # faithful 64-prompt/rollout mini-run
#   SCALE=8b bash trainium/run_pipeline.sh          # Qwen3-8B (see 8B note below)
#
# Hardware note (trn2.3xlarge = 1 Trainium2 chip):
#   A Trainium2 chip has 8 physical NeuronCores, exposed as 4 LOGICAL cores
#   under the default LNC=2 config (verify with `neuron-ls`). Software (this
#   script and torchrun address the 4 logical cores, so NUM_CORES=4 and
#   TP<=4. All sharding happens inside one 96 GiB HBM pool (no data
#   parallelism), which is the binding constraint — see the 8B note.
#
# Key environment overrides:
#   NATIVE_VENV              - Beta-3 native TorchNeuron venv for SFT curation
#   INFER_VENV / TRAIN_VENV  - optional legacy scorer / training venvs
#   SFT_SAMPLES             - SFT prompt count            (default 300000)
#   OPD_STEPS               - Lightning OPD steps         (default 150)
#   NUM_CORES               - logical NeuronCores         (default 4 = trn2.3xlarge)
#   TRAIN_TP / GEN_TP       - tensor-parallel size        (default 4)
#   CUTOFF_LEN              - SFT packed seq length        (default 16384 = paper;
#                             smoke uses 4096. Lower it if step 2 OOMs on 1 chip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Configuration ─────────────────────────────────────────────────────────
SCALE="${SCALE:-4b}"
SMOKE="${SMOKE:-0}"
# Remember whether the user passed SFT_GBS before the SCALE block assigns its
# default, so the smoke override can distinguish "user set it" from "default".
SFT_GBS_USER="${SFT_GBS-}"
# trn2.3xlarge = 1 Trainium2 chip = 4 logical NeuronCores (LNC=2 default).
# For trn1.32xlarge use NUM_CORES=32 TRAIN_TP=8 GEN_TP=8; trn2.48xlarge=64.
NUM_CORES="${NUM_CORES:-4}"
TRAIN_TP="${TRAIN_TP:-4}"          # tensor-parallel size for SFT/OPD (<= NUM_CORES)
# Step 4 teacher scoring backend. "forward" (default) runs a forward pass through
# NeuronModelForCausalLM in the TRAIN venv — the vLLM prompt_logprobs path is
# broken on vllm-neuron 0.16 (see step4_teacher_forward_neuron.py). "vllm" keeps
# the old offline-prompt-logprob path (INFER venv) for SDKs where it works.
TEACHER_BACKEND="${TEACHER_BACKEND:-forward}"
NATIVE_RUNTIME_ENV="${SCRIPT_DIR}/sft_data_generation_native/runtime.env"
if [[ -f "${NATIVE_RUNTIME_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${NATIVE_RUNTIME_ENV}"
fi
NATIVE_VENV="${NATIVE_VENV:-${HOME}/workspace/native_venv}"
[[ -f "${NATIVE_VENV}/bin/activate" ]] || {
    echo "ERROR: Beta-3 native venv not found: ${NATIVE_VENV}" >&2
    echo "Run trainium/sft_data_generation_native/setup_env.sh first." >&2
    exit 1
}
# CUTOFF_LEN is defaulted below to 16384 (paper); the smoke run keeps it (the
# smoke run mirrors the real config, only the dataset and step count shrink).
# Auto-detect the Neuron DLAMI venvs by role — names carry the torch version
# (e.g. _2_9) and the vLLM one is separate (…_inference_vllm_0_16), so match by
# role, not exact name. Override by exporting INFER_VENV / TRAIN_VENV.
# Same logic + rationale as setup_env.sh.
_pick_venv() {  # $1 = grep -E pattern on basename; prints first /opt match
    local d
    for d in /opt/aws_neuronx_venv_*; do
        [[ -e "${d}/bin/activate" ]] || continue
        basename "${d}" | grep -Eq "$1" && { echo "${d}"; return 0; }
    done
    return 1
}
if [[ "${TEACHER_BACKEND}" == "vllm" && -z "${INFER_VENV:-}" ]]; then
    INFER_VENV="$(_pick_venv 'vllm')" || INFER_VENV="$(_pick_venv 'inference')" \
        || { echo "ERROR: no inference venv under /opt; set INFER_VENV manually." >&2; exit 1; }
fi
INFER_VENV="${INFER_VENV:-${NATIVE_VENV}}"
if [[ -z "${TRAIN_VENV:-}" ]]; then
    for d in /opt/aws_neuronx_venv_pytorch_*; do
        case "$(basename "${d}")" in *inference*|*vllm*) continue ;; esac
        [[ -e "${d}/bin/activate" ]] && { TRAIN_VENV="${d}"; break; }
    done
    [[ -n "${TRAIN_VENV:-}" ]] \
        || { echo "ERROR: no training venv under /opt; set TRAIN_VENV manually." >&2; exit 1; }
fi
echo "[run_pipeline] INFER_VENV=${INFER_VENV}"
echo "[run_pipeline] TRAIN_VENV=${TRAIN_VENV}"
echo "[run_pipeline] NATIVE_VENV=${NATIVE_VENV}"

case "${SCALE}" in
    4b)
        STUDENT_BASE="Qwen/Qwen3-4B-Base"
        TEACHER="Qwen/Qwen3-8B"
        TEACHER_TAG="qwen3-8b"
        SFT_GBS="${SFT_GBS:-256}"       # LlamaFactory: 4 x 2 x 32 GPUs
        GEN_TP="${GEN_TP:-${NUM_CORES}}"
        ;;
    8b)
        STUDENT_BASE="Qwen/Qwen3-8B-Base"
        TEACHER="Qwen/Qwen3-32B"
        TEACHER_TAG="qwen3-32b"
        SFT_GBS="${SFT_GBS:-128}"       # LlamaFactory: 2 x 2 x 32 GPUs
        GEN_TP="${GEN_TP:-${NUM_CORES}}"
        ;;
    *)
        echo "SCALE must be 4b or 8b"; exit 1 ;;
esac

# ── Single-chip feasibility guard ─────────────────────────────────────────
# Qwen3-8B full fine-tune needs ~128 GB (bf16 weights + grads + fp32 Adam +
# master weights) which does not fit in one Trainium2 chip's 96 GiB HBM.
# On a single chip TP only splits tensors within the same HBM pool (no data
# parallelism / ZeRO sharding across chips), so raising TP does not help.
# Refusing by default avoids a multi-hour run that OOMs at step 2. A LoRA 8B
# path (which would fit but changes the training method) is not implemented
# yet; ALLOW_8B_OOM=1 forces the full-FT run anyway (expect step 2 to OOM).
if [[ "${SCALE}" == "8b" && "${NUM_CORES}" -le 4 && "${ALLOW_8B_OOM:-0}" != "1" ]]; then
    echo "ERROR: SCALE=8b full fine-tune does not fit on ${NUM_CORES} logical cores"
    echo "       (one Trainium2 chip, 96 GiB). Options:"
    echo "         - 4B scale instead:  SCALE=4b bash trainium/run_pipeline.sh"
    echo "         - full 8B FT on a bigger box: NUM_CORES=32 TRAIN_TP=8 (trn1.32xlarge)"
    echo "       (LoRA-8B-on-one-chip is not wired up yet — ask if you want it.)"
    exit 1
fi

EXTRA_GEN_ARGS=()   # rollout generation (step 3)
SFT_GEN_ARGS=()     # SFT-data generation (step 1)
if [[ "${SMOKE}" == "1" ]]; then
    # Faithful mini-run: SAME config as the real run — generation length
    # (16384), SFT packing length (CUTOFF_LEN=16384), OPD sequence length
    # (5632), learning rates, schedules, warmup, betas are all left at their
    # training values, so the smoke compiles the same Neuron graphs, hits the
    # same memory footprint, and exercises the same code paths as SCALE=4b.
    # Only the DATASET is smaller (64 SFT prompts / 64 rollouts) and the step
    # count is short. The one unavoidable deviation is the global batch size:
    # a 256-sequence batch cannot be formed from 64 samples, so GBS is scaled
    # down — this changes only the gradient-accumulation count, not any
    # compiled shape or hyperparameter. If step 2/5 OOMs here, it will OOM in
    # the real run too (lower CUTOFF_LEN / MAX_SEQ_LEN for both).
    SFT_SAMPLES=64
    SFT_STEPS="${SFT_STEPS:-8}"
    OPD_STEPS="${OPD_STEPS:-8}"
    SFT_GBS="${SFT_GBS_USER:-8}"   # blocks/optimizer step (real run: 256)
    OPD_GBS="${OPD_GBS:-8}"        # rollouts/optimizer step (real run: 256)
    EXTRA_GEN_ARGS=(--num-samples 64)
    export OPD_DEBUG=1             # verbose per-step diagnostics
    echo ">>> SMOKE mode: faithful mini-run — 64 SFT prompts / 64 rollouts, real"
    echo ">>>            config unchanged (cutoff/seq-len/gen-length/LR), GBS"
    echo ">>>            ${SFT_GBS} (SFT) / ${OPD_GBS} (OPD), ${SFT_STEPS} SFT + ${OPD_STEPS} OPD steps, OPD_DEBUG on"
fi
SFT_SAMPLES="${SFT_SAMPLES:-300000}"
OPD_STEPS="${OPD_STEPS:-150}"
# Main run defaults to the paper's SFT packing length. On one trn2.3xlarge
# chip this is the biggest OOM lever for step 2 — if it OOMs, override with
# CUTOFF_LEN=8192 (or 4096); the data is unchanged, only packing efficiency.
CUTOFF_LEN="${CUTOFF_LEN:-16384}"

STUDENT_TAG="qwen3-${SCALE}"
SFT_CKPT_DIR="checkpoints/${STUDENT_TAG}-base-sft-${TEACHER_TAG}"
SFT_HF_DIR="${SFT_CKPT_DIR}-hf"
OPD_CKPT_DIR="checkpoints/${STUDENT_TAG}-lightning-opd"
OPD_HF_DIR="${OPD_CKPT_DIR}-hf"

STATE_DIR="data/.pipeline_state/${SCALE}$( [[ ${SMOKE} == 1 ]] && echo -smoke || true )"
mkdir -p "${STATE_DIR}" data/prompts

stage_done() { [[ -f "${STATE_DIR}/$1.done" ]]; }
mark_done()  { touch "${STATE_DIR}/$1.done"; echo ">>> Stage '$1' complete"; }
run_stage() {  # run_stage <name> <venv> <function>
    local name="$1" venv="$2" fn="$3"
    if stage_done "${name}"; then echo ">>> Stage '${name}' already done, skipping"; return; fi
    echo ">>> Stage '${name}' starting ($(date))"
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    "${fn}"
    deactivate
    mark_done "${name}"
}

consolidate() {  # consolidate <trainer_output_dir> <hf_dir> <tokenizer_src>
    local src="$1" dst="$2" tok="$3"
    mkdir -p "${dst}"
    # optimum-neuron saves sharded (TP) checkpoints; merge into one HF model
    optimum-cli neuron consolidate "${src}" "${dst}"
    # tokenizer + config for vLLM / evaluation
    python3 - "$tok" "$dst" <<'PY'
import sys, shutil, pathlib
tok, dst = sys.argv[1], sys.argv[2]
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(tok, trust_remote_code=True).save_pretrained(dst)
for f in ("config.json", "generation_config.json"):
    p = pathlib.Path(dst) / f
    if not p.exists():
        s = pathlib.Path(tok) / f
        if s.exists():
            shutil.copy(s, p)
PY
}

# ── Stage functions ───────────────────────────────────────────────────────

stage_prompts() {
    python scripts/prepare_sft_prompts.py \
        --hf-dataset open-thoughts/OpenThoughts3-1.2M \
        --output data/prompts/openthoughts3_${SFT_SAMPLES}.jsonl \
        --num-samples "${SFT_SAMPLES}"
    huggingface-cli download zhuzilin/dapo-math-17k \
        --repo-type dataset --include "*.jsonl" \
        --local-dir data/prompts/dapo-math-17k
}

stage_sft_data() {
    if [[ "${NATIVE_VALIDATE:-1}" == "1" ]]; then
        TORCH_LOGS="graph_breaks,recompiles" \
        python -m trainium.sft_data_generation_native.validate_native \
            --model "${TEACHER}" \
            --output agent_artifacts/traces/native_validation.json
    fi
    TEACHER_MODEL="${TEACHER}" \
    SFT_PROMPTS="data/prompts/openthoughts3_${SFT_SAMPLES}.jsonl" \
    OUTPUT_DIR="data/sft_data" \
    NUM_CORES="${NUM_CORES}" TP_SIZE=1 \
    TORCH_LOGS="graph_breaks,recompiles" \
    bash "${SCRIPT_DIR}/step1_generate_sft_data.sh" \
        --mode "${SFT_GEN_MODE:-compile}" \
        --batch-size "${SFT_GEN_BATCH_SIZE:-1}" \
        --prefill-bucket "${SFT_PREFILL_BUCKET:-512}" \
        "${SFT_GEN_ARGS[@]}"
}

stage_sft_merge() {
    python data_curation/merge.py \
        --input-dir data/sft_data \
        --output "data/sft_data/openthoughts3_${SFT_SAMPLES}_${TEACHER_TAG}.parquet" \
        --max-tokens 16384
    find data/sft_data -name "*.arrow" -delete
    rm -rf data/sft_data/rank*
}

stage_sft_train() {
    MODEL_ID="${STUDENT_BASE}" \
    SFT_PARQUET="data/sft_data/openthoughts3_${SFT_SAMPLES}_${TEACHER_TAG}.parquet" \
    OUTPUT_DIR="${SFT_CKPT_DIR}" \
    NUM_CORES="${NUM_CORES}" TP_SIZE="${TRAIN_TP}" \
    GBS="${SFT_GBS}" CUTOFF_LEN="${CUTOFF_LEN}" \
    MAX_STEPS="${SFT_STEPS:-3000}" \
    bash "${SCRIPT_DIR}/step2_sft_train.sh"
}

stage_sft_consolidate() {
    consolidate "${SFT_CKPT_DIR}" "${SFT_HF_DIR}" "${STUDENT_BASE}"
}

stage_rollouts() {
    if [[ "${ROLLOUT_NATIVE_VALIDATE:-1}" == "1" ]]; then
        TORCH_LOGS="graph_breaks,recompiles" \
        python -m trainium.sft_data_generation_native.validate_native \
            --model "${SFT_HF_DIR}" \
            --output agent_artifacts/traces/native_rollout_validation.json
    fi
    SFT_CHECKPOINT="${SFT_HF_DIR}" \
    OPD_PROMPTS="data/prompts/dapo-math-17k/dapo-math-17k.jsonl" \
    OUTPUT_DIR="data/rollouts" \
    NUM_CORES="${NUM_CORES}" TP_SIZE=1 \
    TORCH_LOGS="graph_breaks,recompiles" \
    bash "${SCRIPT_DIR}/step3_collect_rollouts.sh" \
        --mode "${ROLLOUT_GEN_MODE:-compile}" \
        --batch-size "${ROLLOUT_GEN_BATCH_SIZE:-1}" \
        --prefill-bucket "${ROLLOUT_PREFILL_BUCKET:-512}" \
        "${EXTRA_GEN_ARGS[@]}"
}

stage_rollout_merge() {
    python data_curation/merge.py \
        --input-dir data/rollouts \
        --output "data/rollouts/dapo-math-17k-${STUDENT_TAG}-sft-rollouts.parquet"
    find data/rollouts -name "*.arrow" -delete
    rm -rf data/rollouts/rank*
}

stage_teacher_logprobs() {
    TEACHER_MODEL="${TEACHER}" \
    SFT_CHECKPOINT="${SFT_HF_DIR}" \
    ROLLOUT_PARQUET="data/rollouts/dapo-math-17k-${STUDENT_TAG}-sft-rollouts.parquet" \
    OUTPUT_DIR="data/lightning_opd" \
    BACKEND="${TEACHER_BACKEND}" \
    NUM_CORES="${NUM_CORES}" \
    TP_SIZE="${GEN_TP}" \
    bash "${SCRIPT_DIR}/step4_precompute_teacher_logprobs.sh"
}

stage_opd_train() {
    SFT_CHECKPOINT="${SFT_HF_DIR}" \
    LIGHTNING_OPD_DATA="data/lightning_opd/dapo-math-17k-${STUDENT_TAG}-sft-rollouts-lightning-opd-precomputed.parquet" \
    OUTPUT_DIR="${OPD_CKPT_DIR}" \
    NUM_CORES="${NUM_CORES}" TP_SIZE="${TRAIN_TP}" MAX_STEPS="${OPD_STEPS}" \
    GBS="${OPD_GBS:-256}" MAX_SEQ_LEN="${OPD_MAX_SEQ_LEN:-5632}" \
    bash "${SCRIPT_DIR}/step5_lightning_opd_train.sh"
}

stage_opd_consolidate() {
    consolidate "${OPD_CKPT_DIR}" "${OPD_HF_DIR}" "${SFT_HF_DIR}"
}

# ── Pipeline ──────────────────────────────────────────────────────────────
echo "=== Lightning OPD on Trainium: SCALE=${SCALE} student=${STUDENT_BASE} teacher=${TEACHER} ==="

run_stage prompts            "${NATIVE_VENV}" stage_prompts           # step 0
run_stage sft_data           "${NATIVE_VENV}" stage_sft_data           # step 1
run_stage sft_merge          "${NATIVE_VENV}" stage_sft_merge
run_stage sft_train          "${TRAIN_VENV}" stage_sft_train          # step 2
run_stage sft_consolidate    "${TRAIN_VENV}" stage_sft_consolidate
run_stage rollouts           "${NATIVE_VENV}" stage_rollouts           # step 3
run_stage rollout_merge      "${NATIVE_VENV}" stage_rollout_merge
# Step 4 venv depends on the backend: "forward" runs in the TRAIN venv
# (optimum-neuron + torchrun), "vllm" in the INFER venv.
if [[ "${TEACHER_BACKEND}" == "vllm" ]]; then TEACHER_VENV="${INFER_VENV}"; else TEACHER_VENV="${TRAIN_VENV}"; fi
run_stage teacher_logprobs   "${TEACHER_VENV}" stage_teacher_logprobs  # step 4
run_stage opd_train          "${TRAIN_VENV}" stage_opd_train          # step 5
run_stage opd_consolidate    "${TRAIN_VENV}" stage_opd_consolidate    # step 6

echo
echo "=== Pipeline complete ==="
echo "Final model (HF format): ${OPD_HF_DIR}"
