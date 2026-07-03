#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
TAIL_LINES="${TAIL_LINES:-300}"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-qwen35-train}"

job_id="$(
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python - "${RAY_ADDRESS}" <<'PY'
import sys

from ray.job_submission import JobSubmissionClient

address = sys.argv[1]
client = JobSubmissionClient(address)
jobs = client.list_jobs()

if isinstance(jobs, dict):
    items = list(jobs.items())
else:
    items = [(getattr(job, "submission_id", None) or getattr(job, "job_id", None), job) for job in jobs]

def field(job, name, default=None):
    if isinstance(job, dict):
        return job.get(name, default)
    return getattr(job, name, default)

def status(job):
    value = field(job, "status", "")
    return str(value).split(".")[-1].upper()

def ts(item):
    _, job = item
    return field(job, "end_time", None) or field(job, "start_time", None) or 0

failed = [item for item in items if status(item[1]) in {"FAILED", "STOPPED"}]
pool = failed or items
if not pool:
    raise SystemExit("No Ray jobs found.")

latest = max(pool, key=ts)
job_id = latest[0] or field(latest[1], "submission_id", None) or field(latest[1], "job_id", None)
if not job_id:
    raise SystemExit(f"Could not determine Ray job id from: {latest!r}")

print(job_id)
PY
)"

echo "Ray address: ${RAY_ADDRESS}"
echo "Latest failed/stopped job: ${job_id}"
echo
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    ray job logs --address="${RAY_ADDRESS}" "${job_id}" | tail -n "${TAIL_LINES}"
