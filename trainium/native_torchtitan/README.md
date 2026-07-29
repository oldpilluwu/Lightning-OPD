# Native Qwen3-4B SFT on one trn2.48xlarge

This is the native PyTorch/TorchNeuron training path for the Lightning-OPD
Qwen3-4B student. It uses ordinary `torch.nn.Module`, the `neuron` device,
PyTorch FSDP2/DTensor, and the `neuron` process-group backend. It does not use
`torch_xla`, `mark_step`, NxD Training, or `neuronx_distributed`.

The implementation follows the pinned Qwen3 TorchTitan tutorial in the local
`torch-neuronx` repository. The Armin-Neuron examples inform the native
eager/autograd/AdamW smoke test and the optional single-node host-collectives
workaround.

## Topology and training recipe

- Hardware: one `trn2.48xlarge` with 64 logical Neuron devices.
- World size: 64 processes, one process per logical Neuron device.
- Tensor parallelism: TP=4. Qwen3-4B's 32 Q heads and 8 KV heads divide cleanly.
- FSDP2: degree 16 across the remaining mesh dimension.
- Sequence length: 16,384.
- Local batch: 1; global batch: 256; gradient accumulation: 16.
- Optimizer: AdamW, learning rate `8e-5`, cosine schedule, 300 warmup steps.
- Training: 3,000 steps, matching the existing Lightning-OPD SFT config.
- Precision: BF16 parameters during FSDP compute and FP32 reductions.
- Base checkpoint: `Qwen/Qwen3-4B-Base`, loaded from HF safetensors.
- Checkpoints: full resumable state every 100 steps, retaining the newest 10.
- Metrics: W&B logging every step under the original LlamaFactory run name.

The dataloader consumes the generated parquet `messages` column and reproduces
the original LlamaFactory Qwen3 SFT behavior: the explicit Qwen3 template,
empty thinking block insertion when needed, assistant-only labels,
prompt/response-aware truncation, greedy knapsack packing, and position-ID
resets at every packed conversation boundary. The user-visible cutoff remains
16,384 tokens; as in LlamaFactory packing, examples use up to 16,383 tokens and
the final position is padding.

## Put all data on the training instance

The SFT dataset was generated on one `trn2.48xlarge`, and training also uses
one `trn2.48xlarge`. The launcher validates that it can see 300,000 rows and a
`messages` column before starting.

No EFA configuration, cross-node rendezvous, or shared filesystem is needed.
The model, dataset, and output directory only need to be available on the
training host. Use durable EBS or FSx for checkpoints if you need them to
survive loss of the instance; local instance storage is not durable.

Suggested layout:

```text
/data/lightning-opd/
  data/sft/
    openthoughts3_300000_qwen3-8b.parquet
  models/Qwen3-4B-Base/
  runs/
```

## 1. Prepare the native environment

Run inside the TorchNeuron native PyTorch beta environment on the training
instance. The setup helper does not install packages; it checks the native
backend, creates or validates the exact TorchTitan checkout, and applies the
small Qwen3 patch from the local `torch-neuronx` repository.

```bash
export TORCH_NEURONX_SRC="${HOME}/torch-neuronx"
export TORCHTITAN_DIR="${HOME}/torchtitan"
bash trainium/native_torchtitan/setup_torchtitan.sh
```

If the final dependency check fails, install the pinned TorchTitan requirements
in the already-selected beta environment:

```bash
cd "${HOME}/torchtitan"
uv pip install --system -r requirements.txt wandb
```

Log in once with `wandb login`, or set `WANDB_API_KEY` in the environment.
The launcher uses `WANDB_PROJECT=huggingface` and the original LlamaFactory
`run_name` by default. Both can be overridden with environment variables.

Download the complete base model:

```bash
cd "${HOME}/torchtitan"
python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-4B-Base \
  --assets tokenizer safetensors config
```

Move or copy the resulting snapshot to the path used as `MODEL_DIR`, if needed.

## 2. Prove native training on one device

```bash
NEURON_RT_VISIBLE_CORES=0 \
python trainium/native_torchtitan/native_train_smoke.py
```

This exercises a Qwen-like BF16 block, causal SDPA, RMSNorm, SwiGLU, autograd,
and AdamW directly on `device="neuron"`. The first step compiles native kernels;
later steps reuse them. Continue only after the loss is finite, decreases, and
the script prints `NATIVE_TRAIN_SMOKE_OK`.

## 3. Run a 64-device topology smoke test

```bash
SFT_DATA=/data/lightning-opd/data/sft \
MODEL_DIR=/data/lightning-opd/models/Qwen3-4B-Base \
OUTPUT_ROOT=/data/lightning-opd/runs \
RUN_ID=qwen3-4b-native-smoke \
SMOKE=1 \
bash trainium/native_torchtitan/run_sft_trn2_48xlarge.sh
```

This performs two full-model eager training steps with TP=4, FSDP=16, and
global batch 16, so the smoke test has one microbatch per FSDP rank and no
gradient accumulation. Checkpoint loading remains enabled so it starts from
Qwen3-4B-Base, while `checkpoint.load_only` prevents smoke outputs from being
saved. The first step can take several minutes while native kernels compile.
Require finite loss on both steps.

The launcher defaults `TORCH_NEURONX_ENABLE_HOST_CC=0`, matching the official
single-node Trn2 Qwen3 TorchTitan recipe. If the smoke test hangs during a
barrier or reports an OFI/process-group initialization failure, retry it once
with the Armin-Neuron single-node workaround:

```bash
TORCH_NEURONX_ENABLE_HOST_CC=1 \
SFT_DATA=/data/lightning-opd/data/sft \
MODEL_DIR=/data/lightning-opd/models/Qwen3-4B-Base \
OUTPUT_ROOT=/data/lightning-opd/runs \
RUN_ID=qwen3-4b-native-smoke-host-cc \
SMOKE=1 \
bash trainium/native_torchtitan/run_sft_trn2_48xlarge.sh
```

## 4. Start or resume the 3,000-step run

```bash
SFT_DATA=/data/lightning-opd/data/sft \
MODEL_DIR=/data/lightning-opd/models/Qwen3-4B-Base \
OUTPUT_ROOT=/data/lightning-opd/runs \
RUN_ID=qwen3-4b-native-sft \
bash trainium/native_torchtitan/run_sft_trn2_48xlarge.sh
```

On first launch, TorchTitan loads the HF base weights. On a later launch with
the same `RUN_ID`, it resumes the newest DCP checkpoint. The launcher refuses
an empty checkpoint directory because TorchTitan could otherwise skip HF
initialization.

## Compile phase

The production config keeps `[compile].enable=false`, matching the native Qwen3
training example in `torch-neuronx`. First establish correct, stable eager
training. After that gate passes, copy the TOML, set `enable=true` with
`backend="neuron"`, and use a new two-step smoke `RUN_ID`. Do not switch the
production run until loss and checkpoint behavior match the eager smoke and the
repeated transformer block has no graph breaks.

## Expected outputs

For `RUN_ID=qwen3-4b-native-sft`:

```text
/data/lightning-opd/runs/qwen3-4b-native-sft/
  checkpoint/
```

All checkpoints, including the final checkpoint, contain model, optimizer,
scheduler, dataloader, and training state. This matches the original
LlamaFactory `save_only_model: false` resume behavior. TorchTitan stores these
as distributed checkpoints rather than Hugging Face safetensors; export to HF
format is a separate post-training operation.
