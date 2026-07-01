#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SFT_PROBE_METRICS_DIR="${SFT_PROBE_METRICS_DIR:-${EXP_DIR}/sft_probe_metrics}"
SFT_PROBE_POLL_SECONDS="${SFT_PROBE_POLL_SECONDS:-60}"
SFT_PROBE_KEEP_LATEST="${SFT_PROBE_KEEP_LATEST:-1}"
SFT_PROBE_PLATEAU_THRESHOLD="${SFT_PROBE_PLATEAU_THRESHOLD:-0.01}"
SFT_PROBE_PLATEAU_PATIENCE="${SFT_PROBE_PLATEAU_PATIENCE:-3}"

python scripts/qwen35_2b_9b/monitor_sft_saturation.py \
    --checkpoint-dir "${SFT_OUTPUT_DIR}" \
    --probe-parquet "${SFT_PROBE_PRECOMPUTED}" \
    --output-dir "${SFT_PROBE_METRICS_DIR}" \
    --watch \
    --poll-seconds "${SFT_PROBE_POLL_SECONDS}" \
    --keep-latest "${SFT_PROBE_KEEP_LATEST}" \
    --plateau-threshold "${SFT_PROBE_PLATEAU_THRESHOLD}" \
    --plateau-patience "${SFT_PROBE_PLATEAU_PATIENCE}" \
    ${SFT_PROBE_NO_PRUNE:+--no-prune}
