#!/usr/bin/env bash
# Full-parameter Qwen3-4B-Base SFT on one trn2.48xlarge.

set -euo pipefail

: "${TOKENIZED_DATA:?Set TOKENIZED_DATA to prepare_sft_dataset.py output}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16384}"
SMOKE="${SMOKE:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type transformer --retry_failed_compilation}"
export NEURON_FUSE_SOFTMAX="${NEURON_FUSE_SOFTMAX:-1}"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS="${NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS:-3}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-64}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${SMOKE}" == "1" ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    TP_DEGREE="${TP_DEGREE:-8}"
    MAX_STEPS="${MAX_STEPS:-2}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    REPORT_TO="${REPORT_TO:-none}"
elif [[ "${NEURON_EXTRACT_GRAPHS_ONLY:-0}" == "1" ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-64}"
    TP_DEGREE="${TP_DEGREE:-8}"
    MAX_STEPS="${MAX_STEPS:-2}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    REPORT_TO="${REPORT_TO:-none}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-64}"
    TP_DEGREE="${TP_DEGREE:-8}"
    MAX_STEPS="${MAX_STEPS:-3000}"
    # Original global batch: 32 GPU DP ranks * 4 microbatch * 2 accumulation = 256.
    # Trainium: 64 processes / TP8 = 8 DP ranks; 8 * 1 * 32 = the same 256.
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
    SAVE_STEPS="${SAVE_STEPS:-100}"
    REPORT_TO="${REPORT_TO:-wandb}"
fi

if (( NPROC_PER_NODE % TP_DEGREE != 0 )); then
    echo "ERROR: NPROC_PER_NODE must be divisible by TP_DEGREE." >&2
    exit 1
fi

DP_DEGREE=$((NPROC_PER_NODE / TP_DEGREE))
GLOBAL_BATCH=$((DP_DEGREE * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [[ "${SMOKE}" != "1" && "${GLOBAL_BATCH}" -ne 256 ]]; then
    echo "ERROR: Production global batch must remain 256; got ${GLOBAL_BATCH}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "=== Optimum Neuron Qwen3 full SFT ==="
echo "Model:          ${MODEL_ID}"
echo "Dataset:        ${TOKENIZED_DATA}"
echo "Output:         ${OUTPUT_DIR}"
echo "Processes:      ${NPROC_PER_NODE}"
echo "TP / DP:        ${TP_DEGREE} / ${DP_DEGREE}"
echo "Sequence:       ${SEQUENCE_LENGTH}"
echo "Steps:          ${MAX_STEPS}"
echo "Microbatch:     ${PER_DEVICE_BATCH_SIZE}"
echo "Grad accum:     ${GRADIENT_ACCUMULATION_STEPS}"
echo "Global batch:   ${GLOBAL_BATCH}"
echo "Trainable mode: FULL MODEL (no PEFT/LoRA)"

exec torchrun \
    --nnodes 1 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:0" \
    "${SCRIPT_DIR}/sft_qwen3.py" \
    --model_id "${MODEL_ID}" \
    --tokenized_dataset "${TOKENIZED_DATA}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --do_train true \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --gradient_checkpointing false \
    --learning_rate 8e-5 \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --stochastic_rounding_enabled false \
    --tensor_parallel_size "${TP_DEGREE}" \
    --disable_sequence_parallel false \
    --zero_1 false \
    --async_save false \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 10 \
    --save_only_model false \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataloader_pin_memory true \
    --ddp_timeout 180000000 \
    --seed 42 \
    --data_seed 42 \
    --report_to "${REPORT_TO}" \
    --run_name qwen3-4b-base-open-thoughts3-qwen3-8b \
    --overwrite_output_dir false \
    "${EXTRA_ARGS[@]}"
