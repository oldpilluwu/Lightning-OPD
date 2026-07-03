#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
TAIL_LINES="${TAIL_LINES:-300}"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-qwen35-train}"

tmp_json="$(mktemp)"
trap 'rm -f "${tmp_json}"' EXIT

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    ray job list --address="${RAY_ADDRESS}" --format=json > "${tmp_json}"

job_id="$(
    python - "${tmp_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    jobs = json.load(f)

if isinstance(jobs, dict):
    jobs = list(jobs.values())

def ts(job):
    return job.get("end_time") or job.get("start_time") or 0

failed = [
    job
    for job in jobs
    if str(job.get("status", "")).upper() in {"FAILED", "STOPPED"}
]
pool = failed or jobs
if not pool:
    raise SystemExit("No Ray jobs found.")

latest = max(pool, key=ts)
print(latest.get("job_id") or latest.get("submission_id") or latest.get("id"))
PY
)"

echo "Ray address: ${RAY_ADDRESS}"
echo "Latest failed/stopped job: ${job_id}"
echo
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    ray job logs --address="${RAY_ADDRESS}" "${job_id}" | tail -n "${TAIL_LINES}"
