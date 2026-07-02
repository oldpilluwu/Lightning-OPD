#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python - <<'PY'
from pathlib import Path

qwen_sft = Path("scripts/qwen35_2b_9b/02_run_sft.sh")
qwen_sft.write_text("""#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-500}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"
SFT_CUTOFF_LEN="${SFT_CUTOFF_LEN:-8192}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
SFT_DEEPSPEED_CONFIG="${SFT_DEEPSPEED_CONFIG:-}"

mkdir -p "${SFT_OUTPUT_DIR}"

if command -v llamafactory-cli >/dev/null 2>&1; then
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

SFT_ARGS=(
    --model_name_or_path "${STUDENT_BASE}" \\
    --stage sft \\
    --do_train true \\
    --finetuning_type full \\
    --packing true \\
    --dataset qwen35_2b9b_sft \\
    --dataset_dir configs/sft \\
    --template qwen3 \\
    --cutoff_len "${SFT_CUTOFF_LEN}" \\
    --overwrite_cache true \\
    --preprocessing_num_workers 8 \\
    --dataloader_persistent_workers true \\
    --dataloader_pin_memory true \\
    --dataloader_num_workers 2 \\
    --output_dir "${SFT_OUTPUT_DIR}" \\
    --logging_steps 1 \\
    --save_steps "${SFT_SAVE_STEPS}" \\
    --save_total_limit "${SFT_SAVE_TOTAL_LIMIT}" \\
    --plot_loss true \\
    --overwrite_output_dir false \\
    --save_only_model false \\
    --report_to none \\
    --run_name qwen35-2b-base-open-thoughts3-qwen35-9b \\
    --per_device_train_batch_size "${SFT_BATCH_SIZE}" \\
    --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \\
    --learning_rate 8e-5 \\
    --max_steps "${SFT_MAX_STEPS}" \\
    --lr_scheduler_type cosine \\
    --warmup_ratio 0.1 \\
    --bf16 true \\
    --ddp_timeout 180000000
)

if [[ -n "${SFT_DEEPSPEED_CONFIG}" ]]; then
    SFT_ARGS+=(--deepspeed "${SFT_DEEPSPEED_CONFIG}")
fi

"${LLAMAFACTORY_TRAIN[@]}" "${SFT_ARGS[@]}"

echo "SFT output: ${SFT_OUTPUT_DIR}"
""")

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
PY

chmod +x scripts/qwen35_2b_9b/02_run_sft.sh
echo "SFT launcher now passes explicit llamafactory-cli arguments."
