#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Lightning OPD on AWS Trainium — single resumable entrypoint.
#
# Runs the full pipeline (steps 0-6) for one scale on one trn1.32xlarge /
# trn2.48xlarge instance. Every stage writes a marker into
# data/.pipeline_state/ when it completes; re-running the script resumes at
# the first unfinished stage (and the generation / scoring stages also resume
# mid-stage from their own checkpoints).
#
# Usage:
#   SCALE=4b bash trainium/run_pipeline.sh          # Qwen3-4B-Base + Qwen3-8B teacher
#   SCALE=8b bash trainium/run_pipeline.sh          # Qwen3-8B-Base + Qwen3-32B teacher
#   SCALE=4b SMOKE=1 bash trainium/run_pipeline.sh  # tiny end-to-end test (5k SFT / 2k OPD)
#
# Key environment overrides:
#   INFER_VENV / TRAIN_VENV - Neuron venv paths (see setup_env.sh)
#   SFT_SAMPLES             - SFT prompt count      (default 300000)
#   OPD_STEPS               - Lightning OPD steps   (default 150)
#   NUM_CORES               - NeuronCores available (default 32)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Configuration ─────────────────────────────────────────────────────────
SCALE="${SCALE:-4b}"
SMOKE="${SMOKE:-0}"
NUM_CORES="${NUM_CORES:-32}"
INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_7_nxd_inference}"
TRAIN_VENV="${TRAIN_VENV:-/opt/aws_neuronx_venv_pytorch_2_7}"

case "${SCALE}" in
    4b)
        STUDENT_BASE="Qwen/Qwen3-4B-Base"
        TEACHER="Qwen/Qwen3-8B"
        TEACHER_TAG="qwen3-8b"
        SFT_GBS="${SFT_GBS:-256}"       # LlamaFactory: 4 x 2 x 32 GPUs
        GEN_TP="${GEN_TP:-8}"
        ;;
    8b)
        STUDENT_BASE="Qwen/Qwen3-8B-Base"
        TEACHER="Qwen/Qwen3-32B"
        TEACHER_TAG="qwen3-32b"
        SFT_GBS="${SFT_GBS:-128}"       # LlamaFactory: 2 x 2 x 32 GPUs
        GEN_TP="${GEN_TP:-8}"
        ;;
    *)
        echo "SCALE must be 4b or 8b"; exit 1 ;;
esac

EXTRA_GEN_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
    # Tiny end-to-end validation run: 5k SFT prompts, 2k OPD rollouts
    SFT_SAMPLES=5000
    SFT_STEPS=100
    OPD_STEPS=20
    EXTRA_GEN_ARGS=(--num-samples 2000)
    echo ">>> SMOKE mode: 5k SFT prompts / 100 SFT steps, 2k rollouts, 20 OPD steps"
fi
SFT_SAMPLES="${SFT_SAMPLES:-300000}"
OPD_STEPS="${OPD_STEPS:-150}"

STUDENT_TAG="qwen3-${SCALE}"
SFT_CKPT_DIR="checkpoints/${STUDENT_TAG}-base-sft-${TEACHER_TAG}"
SFT_HF_DIR="${SFT_CKPT_DIR}-hf"
OPD_CKPT_DIR="checkpoints/${STUDENT_TAG}-lightning-opd"
OPD_HF_DIR="${OPD_CKPT_DIR}-hf"

OPD_PROBE_PARQUET="data/probe/opd_probe_${TEACHER_TAG}.parquet"

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

stage_opd_probe() {
    # Frozen probe set: teacher responses + logprobs on OPD prompts, used to
    # monitor student convergence during SFT (fwd_kl, drift) and decide when
    # SFT is enough to move on to OPD. See trainium/opd_probe.py.
    python "${SCRIPT_DIR}/build_opd_probe.py" \
        --teacher-model "${TEACHER}" \
        --opd-prompts data/prompts/dapo-math-17k/dapo-math-17k.jsonl \
        --output "${OPD_PROBE_PARQUET}" \
        --num-prompts "${PROBE_SIZE:-64}" \
        --tensor-parallel-size "${GEN_TP}"
}

stage_sft_data() {
    TEACHER_MODEL="${TEACHER}" \
    SFT_PROMPTS="data/prompts/openthoughts3_${SFT_SAMPLES}.jsonl" \
    OUTPUT_DIR="data/sft_data" \
    NUM_CORES="${NUM_CORES}" TP_SIZE="${GEN_TP}" \
    bash "${SCRIPT_DIR}/step1_generate_sft_data.sh"
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
    NUM_CORES="${NUM_CORES}" GBS="${SFT_GBS}" \
    MAX_STEPS="${SFT_STEPS:-3000}" \
    OPD_PROBE="${OPD_PROBE_PARQUET}" \
    PROBE_EVERY="${PROBE_EVERY:-5}" PROBE_SIZE="${PROBE_SIZE:-64}" \
    bash "${SCRIPT_DIR}/step2_sft_train.sh"
}

stage_sft_consolidate() {
    consolidate "${SFT_CKPT_DIR}" "${SFT_HF_DIR}" "${STUDENT_BASE}"
}

stage_rollouts() {
    SFT_CHECKPOINT="${SFT_HF_DIR}" \
    OPD_PROMPTS="data/prompts/dapo-math-17k/dapo-math-17k.jsonl" \
    OUTPUT_DIR="data/rollouts" \
    NUM_CORES="${NUM_CORES}" TP_SIZE="${GEN_TP}" \
    bash "${SCRIPT_DIR}/step3_collect_rollouts.sh" "${EXTRA_GEN_ARGS[@]}"
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
    TP_SIZE="${GEN_TP}" \
    bash "${SCRIPT_DIR}/step4_precompute_teacher_logprobs.sh"
}

stage_opd_train() {
    SFT_CHECKPOINT="${SFT_HF_DIR}" \
    LIGHTNING_OPD_DATA="data/lightning_opd/dapo-math-17k-${STUDENT_TAG}-sft-rollouts-lightning-opd-precomputed.parquet" \
    OUTPUT_DIR="${OPD_CKPT_DIR}" \
    NUM_CORES="${NUM_CORES}" MAX_STEPS="${OPD_STEPS}" \
    OPD_PROBE="${OPD_PROBE_PARQUET}" \
    PROBE_EVERY="${PROBE_EVERY:-5}" PROBE_SIZE="${PROBE_SIZE:-64}" \
    bash "${SCRIPT_DIR}/step5_lightning_opd_train.sh"
}

stage_opd_consolidate() {
    consolidate "${OPD_CKPT_DIR}" "${OPD_HF_DIR}" "${SFT_HF_DIR}"
}

# ── Pipeline ──────────────────────────────────────────────────────────────
echo "=== Lightning OPD on Trainium: SCALE=${SCALE} student=${STUDENT_BASE} teacher=${TEACHER} ==="

run_stage prompts            "${INFER_VENV}" stage_prompts            # step 0
run_stage opd_probe          "${INFER_VENV}" stage_opd_probe          # probe set for SFT monitoring
run_stage sft_data           "${INFER_VENV}" stage_sft_data           # step 1
run_stage sft_merge          "${INFER_VENV}" stage_sft_merge
run_stage sft_train          "${TRAIN_VENV}" stage_sft_train          # step 2
run_stage sft_consolidate    "${TRAIN_VENV}" stage_sft_consolidate
run_stage rollouts           "${INFER_VENV}" stage_rollouts           # step 3
run_stage rollout_merge      "${INFER_VENV}" stage_rollout_merge
run_stage teacher_logprobs   "${INFER_VENV}" stage_teacher_logprobs   # step 4
run_stage opd_train          "${TRAIN_VENV}" stage_opd_train          # step 5
run_stage opd_consolidate    "${TRAIN_VENV}" stage_opd_consolidate    # step 6

echo
echo "=== Pipeline complete ==="
echo "Final model (HF format): ${OPD_HF_DIR}"
