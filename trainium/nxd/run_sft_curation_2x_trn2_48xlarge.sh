#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Two-instance launcher for the validated Qwen3-8B layout:
#   2 x trn2.48xlarge
#   16 replicas per instance
#   TP=4 and max_num_seqs=16 per replica
#
# Run this script on both instances. Set INSTANCE_RANK=0 on the first and
# INSTANCE_RANK=1 on the second. Before launch, download the matching 150K
# prompt shard into WORK_ROOT/prompts. No network rendezvous is needed.

set -euo pipefail

if [[ ! "${INSTANCE_RANK:-}" =~ ^[01]$ ]]; then
    echo "ERROR: set INSTANCE_RANK=0 on the first instance or INSTANCE_RANK=1 on the second." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_INSTANCES=2
export NUM_WORKERS=16
export TP_SIZE=4
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export PRE_SHARDED_INPUT=1

exec bash "${SCRIPT_DIR}/run_sft_curation_trn2_48xlarge.sh"
