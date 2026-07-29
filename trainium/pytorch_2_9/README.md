# Qwen3-4B full SFT with direct PyTorch 2.9 on one Trn2

This implementation uses the AWS DLAMI
`aws_neuronx_venv_pytorch_2_9`, Torch/XLA, and the DLAMI-provided NxD Core
0.19 package directly. It contains no Optimum code or dependency.

The model is a normal `torch.nn.Module` composed of NxD tensor-parallel
embeddings and linear layers. Attention uses the NxD NKI flash-attention
kernel. Training, optimizer stepping, scheduling, checkpointing, and data
loading are implemented directly in PyTorch.

## Reproduction settings

The production run preserves the original LLaMA-Factory SFT contract:

- `Qwen/Qwen3-4B-Base`
- Full-parameter training; no LoRA or PEFT
- LLaMA-Factory `qwen3` chat template with thinking enabled
- Prompt labels masked with `-100`
- `packing: true`, `neat_packing: false`
- Fixed sequence length 16,384
- AdamW, learning rate `8e-5`, weight decay `0`
- Betas `0.9/0.999`, epsilon `1e-8`, gradient clipping `1.0`
- Cosine schedule, warmup ratio `0.1`
- 3,000 optimizer steps
- BF16, stochastic rounding disabled
- ZeRO-0 optimizer state, with NxD-only FP32 master-weight and FP32
  gradient-accumulation modes disabled (NxD 0.19 supports those modes only
  with ZeRO-1)
- Sequence parallelism enabled (matching `disable_sequence_parallel: false`)
- Gradient checkpointing disabled
- Save every 100 steps and keep 10
- Seed and data seed 42
- W&B reporting in production
- DDP timeout 180,000,000 seconds
- Dataset cache overwrite and non-overwriting training output
- Original global batch 256

On one `trn2.48xlarge`, production launches 64 LNC=2 workers with TP=8 and
DP=8. Microbatch 1 and gradient accumulation 32 give global batch 256.

## Files

- `modeling_qwen3_nxd.py`: direct tensor-parallel Qwen3 implementation
- `convert_qwen3_checkpoint.py`: explicit PyTorch TP=8 checkpoint sharding
- `train_sft.py`: direct PyTorch/XLA training loop
- `llamafactory_compat.py`: exact preprocessing behavior
- `prepare_sft_dataset.py`: preprocessing and packing
- `run_smoke_test.sh`: two-step full-model hardware test
- `run_sft_trn2_48xlarge.sh`: production launcher

## 1. Synchronize the new implementation

The old directory must not remain on the instance:

```bash
cd "$HOME/Lightning-OPD"
test ! -d trainium/optimum_neuron
test -d trainium/pytorch_2_9
```

## 2. Set up the PyTorch 2.9 environment

```bash
cd "$HOME/Lightning-OPD"
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
unset PYTHONPATH

bash trainium/pytorch_2_9/setup_instance.sh
```

The setup removes the obsolete
`$HOME/optimum-neuron-qwen3-pt29` checkout and uninstalls the `optimum` and
`optimum-neuron` Python packages. It also removes any ignored bytecode left in
the deleted `trainium/optimum_neuron` directory. It never changes Torch, Torch-XLA,
Torch-NeuronX, NxD, libneuronxla, or neuronx-cc. Success ends with:

```text
PYTORCH_2_9_DIRECT_SETUP_OK
```

## 3. Smoke test

The smoke test automatically uses:

```text
$HOME/lightning-opd-prep/sft_data/openthoughts3_300000_qwen3-8b.parquet
```

It downloads Qwen3 once, creates direct TP=8 checkpoint shards, preprocesses
1,000 source rows at sequence length 2,048, and runs two full-model optimizer
steps. The NKI flash-attention kernel requires sequence lengths in multiples
of 2,048.

```bash
cd "$HOME/Lightning-OPD"
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
unset PYTHONPATH

bash trainium/pytorch_2_9/run_smoke_test.sh
```

Success ends with:

```text
TRAINING_OK
SMOKE_TEST_OK
```

## 4. Prepare the full production dataset

```bash
export REPO_DIR="$HOME/Lightning-OPD"
export PREP_DIR="$HOME/lightning-opd-prep"
export SFT_DATA="$PREP_DIR/sft_data/openthoughts3_300000_qwen3-8b.parquet"
export TOKENIZED_DATA="$PREP_DIR/sft_data/qwen3-4b-lf-packed-16384"
export MODEL_CHECKPOINT="$PREP_DIR/model_checkpoints/qwen3-4b-base-tp8"
export OUTPUT_ROOT="$PREP_DIR/training_runs"

cd "$REPO_DIR"

python trainium/pytorch_2_9/prepare_sft_dataset.py \
  --dataset-path "$SFT_DATA" \
  --output-dir "$TOKENIZED_DATA" \
  --model-id Qwen/Qwen3-4B-Base \
  --cutoff-len 16384 \
  --expected-rows 300000 \
  --preprocessing-num-workers 16 \
  --preprocessing-batch-size 1000 \
  --overwrite
```

Create the model checkpoint if the smoke test did not already create it:

```bash
MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
  bash trainium/pytorch_2_9/prepare_model_checkpoint.sh
```

## 5. Precompile

```bash
mkdir -p "$OUTPUT_ROOT"

TOKENIZED_DATA="$TOKENIZED_DATA" \
MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
OUTPUT_DIR="$OUTPUT_ROOT/qwen3-4b-full-sft-compile" \
SEQUENCE_LENGTH=16384 \
MAX_STEPS=2 \
REPORT_TO=none \
neuron_parallel_compile \
  bash trainium/pytorch_2_9/run_sft_trn2_48xlarge.sh \
  2>&1 | tee "$OUTPUT_ROOT/qwen3-4b-full-sft-compile.log"
```

## 6. Train

Authenticate W&B first if production metrics should be uploaded:

```bash
wandb login
```

Then launch:

```bash
TOKENIZED_DATA="$TOKENIZED_DATA" \
MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
OUTPUT_DIR="$OUTPUT_ROOT/qwen3-4b-full-sft" \
SEQUENCE_LENGTH=16384 \
bash trainium/pytorch_2_9/run_sft_trn2_48xlarge.sh \
  2>&1 | tee "$OUTPUT_ROOT/qwen3-4b-full-sft.log"
```

Resume from step 100:

```bash
TOKENIZED_DATA="$TOKENIZED_DATA" \
MODEL_CHECKPOINT="$MODEL_CHECKPOINT" \
OUTPUT_DIR="$OUTPUT_ROOT/qwen3-4b-full-sft" \
SEQUENCE_LENGTH=16384 \
RESUME_TAG=step_100 \
bash trainium/pytorch_2_9/run_sft_trn2_48xlarge.sh
```

NxD checkpoints are already sharded by TP rank and include the optimizer and
scheduler state. No adapter merge exists.
