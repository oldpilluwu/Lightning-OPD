#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python - <<'PY'
from pathlib import Path

replacements = {
    Path("scripts/qwen35_2b_9b/02_run_sft.sh"): '''torchrun \\
    --nnodes "${NUM_NODES}" \\
    --nproc_per_node="${NUM_GPUS}" \\
    --rdzv_id "${RANDOM}" \\
    --rdzv_backend c10d \\
    --rdzv_endpoint "${MASTER_ADDR}:29500" \\
    -m llamafactory.cli.train \\
''',
    Path("configs/sft/run_sft.sh"): '''torchrun \\
    --nnodes "${NUM_NODES}" \\
    --nproc_per_node="${NUM_GPUS}" \\
    --rdzv_id $RANDOM \\
    --rdzv_backend c10d \\
    --rdzv_endpoint "${MASTER_ADDR}:29500" \\
    -m llamafactory.cli.train \\
''',
}

new_block = '''if command -v llamafactory-cli >/dev/null 2>&1; then
    LLAMAFACTORY_TRAIN=(llamafactory-cli train)
else
    LLAMAFACTORY_TRAIN=(python -m llamafactory.cli train)
fi

if [[ "${NUM_NODES}" != "1" || "${NUM_GPUS}" != "1" ]]; then
    export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
    export NNODES="${NUM_NODES}"
    export NPROC_PER_NODE="${NUM_GPUS}"
    export MASTER_ADDR="${MASTER_ADDR}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
fi

"${LLAMAFACTORY_TRAIN[@]}" \\
'''

for path, old_block in replacements.items():
    text = path.read_text()
    if old_block in text:
        path.write_text(text.replace(old_block, new_block))
        print(f"patched {path}")
    elif "LLAMAFACTORY_TRAIN=(llamafactory-cli train)" in text:
        print(f"already patched {path}")
    else:
        raise SystemExit(f"could not find expected launcher block in {path}")
PY

echo "SFT launcher now uses llamafactory-cli train."
