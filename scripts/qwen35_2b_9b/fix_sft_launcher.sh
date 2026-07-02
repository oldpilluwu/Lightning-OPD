#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python - <<'PY'
from pathlib import Path
import json

dataset_info = Path("configs/sft/dataset_info.json")
info = json.loads(dataset_info.read_text())
info["qwen35_2b9b_sft"]["file_name"] = "../../data/qwen35_2b_9b/sft_data/train_llamafactory.parquet"
dataset_info.write_text(json.dumps(info, indent=2) + "\n")

qwen_sft = Path("scripts/qwen35_2b_9b/02_run_sft.sh")
native_runner = Path("scripts/qwen35_2b_9b/run_sft_native.py")
if not native_runner.exists():
    raise SystemExit("Missing native SFT runner. Sync scripts/qwen35_2b_9b/run_sft_native.py to the remote first.")
if "SFT_BACKEND" not in qwen_sft.read_text():
    raise SystemExit("02_run_sft.sh is still the old LLaMA-Factory launcher. Sync the latest file to the remote first.")

generic_sft = Path("configs/sft/run_sft.sh")
text = generic_sft.read_text()
old_blocks = [
    '''torchrun \\
    --nnodes "${NUM_NODES}" \\
    --nproc_per_node="${NUM_GPUS}" \\
    --rdzv_id $RANDOM \\
    --rdzv_backend c10d \\
    --rdzv_endpoint "${MASTER_ADDR}:29500" \\
    -m llamafactory.cli.train \\
''',
]
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
for old_block in old_blocks:
    if old_block in text:
        text = text.replace(old_block, new_block)
generic_sft.write_text(text)

print(f"rewrote {qwen_sft}")
print(f"checked {generic_sft}")
print(f"patched {dataset_info}")
PY

chmod +x scripts/qwen35_2b_9b/02_run_sft.sh
echo "SFT launcher now passes explicit llamafactory-cli arguments."
