#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compatibility wrapper. The implementation is pure native PyTorch.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/sft_data_generation_native/run_curation_native.sh" "$@"
