#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${OPD_CHECKPOINT_ROOT:?Set OPD_CHECKPOINT_ROOT to the Megatron checkpoint root}"
: "${SFT_CHECKPOINT:=/root/models/Qwen3-1.7B-SFT}"
: "${TEACHER_CHECKPOINT:=/root/models/Qwen3-4B}"
: "${OPD_HF_CHECKPOINT_ROOT:=/root/models/Qwen3-1.7B-SFT-Qwen3-4B-OPD-hf}"
: "${OPD_DATA_DIR:=/root/datasets/qwen3-1.7b-opd}"
: "${OPD_RESULTS_DIR:=results/qwen3-1.7b-a6000-opd}"

python tools/run_opd_post_training.py all \
  --checkpoint-root "${OPD_CHECKPOINT_ROOT}" \
  --origin-hf "${SFT_CHECKPOINT}" \
  --hf-root "${OPD_HF_CHECKPOINT_ROOT}" \
  --teacher-model "${TEACHER_CHECKPOINT}" \
  --data-dir "${OPD_DATA_DIR}" \
  --output-dir "${OPD_RESULTS_DIR}"
