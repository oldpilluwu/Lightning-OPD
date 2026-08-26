#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPD_PROJECT_ROOT:=$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${OPD_CHECKPOINT_ROOT:=${OPD_PROJECT_ROOT}/models/Qwen3-1.7B-SFT-Qwen3-4B-OPD_ckpt__qwen3-1.7b-a6000-opd}"
: "${SFT_CHECKPOINT:=${OPD_PROJECT_ROOT}/models/Qwen3-1.7B-SFT}"
: "${TEACHER_CHECKPOINT:=${OPD_PROJECT_ROOT}/models/Qwen3-4B}"
: "${OPD_HF_CHECKPOINT_ROOT:=${OPD_PROJECT_ROOT}/models/Qwen3-1.7B-SFT-Qwen3-4B-OPD-hf}"
: "${OPD_DATA_DIR:=${OPD_PROJECT_ROOT}/datasets/qwen3-1.7b-opd}"
: "${OPD_RESULTS_DIR:=${OPD_PROJECT_ROOT}/results/qwen3-1.7b-a6000-opd}"

python tools/run_opd_post_training.py all \
  --checkpoint-root "${OPD_CHECKPOINT_ROOT}" \
  --origin-hf "${SFT_CHECKPOINT}" \
  --hf-root "${OPD_HF_CHECKPOINT_ROOT}" \
  --teacher-model "${TEACHER_CHECKPOINT}" \
  --data-dir "${OPD_DATA_DIR}" \
  --output-dir "${OPD_RESULTS_DIR}"
