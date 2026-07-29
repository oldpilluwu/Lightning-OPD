# Qwen3-4B full SFT on one trn2.48xlarge

This is full-parameter training. There is no PEFT configuration, no adapter,
and no LoRA merge step. The training backend is the maintained
Optimum Neuron/XLA stack and its custom Qwen3 implementation.

## Reproduction contract

The original configuration is
`configs/sft/qwen3-4b-base-open-thoughts3-qwen3-8b.yaml`.

| Original setting | Trainium reproduction |
|---|---|
| `Qwen/Qwen3-4B-Base` | identical |
| `finetuning_type: full` | every model parameter is trainable; startup aborts otherwise |
| `template: qwen3` | identical formatter and reasoning-template behavior |
| user-token loss masking | identical: source labels are `-100` |
| `cutoff_len: 16384` | identical 16,384-token tensors |
| `packing: true` | identical LLaMA-Factory greedy packing |
| `neat_packing: false` default | identical cross-example attention and reset position IDs |
| `learning_rate: 8e-5` | identical |
| `max_steps: 3000` | identical |
| cosine schedule, 10% warmup | identical |
| BF16 | identical |
| stochastic rounding | disabled (the original did not enable it) |
| logging every step | identical |
| checkpoint every 100 steps, retain 10 | identical |
| W&B run name | identical |
| effective global batch 256 | identical |
| DeepSpeed ZeRO-0 | `zero_1=false`, so optimizer state is not DP-sharded |

There are two unavoidable hardware decompositions:

1. The GPU run used 32 data-parallel replicas with microbatch 4 and accumulation
   2. One trn2.48xlarge uses 64 processes, TP8, and therefore DP8. The launcher
   uses microbatch 1 and accumulation 32, preserving the optimizer batch of 256
   at every update.
2. `enable_liger_kernel` is a CUDA kernel switch. On Trainium it is replaced by
   Optimum Neuron's Qwen3 model, FlashAttention2 implementation, tensor
   parallelism, and sequence parallelism. It cannot be enabled literally on
   Neuron.

All model, data, loss, optimizer, scheduler, logging, and checkpoint semantics
remain the original ones. Tensor parallelism changes placement, not the trained
parameter set.

## Why preprocessing is not `apply_chat_template`

The implementation mirrors LLaMA-Factory commit
`9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d`.

For each conversation it:

1. Renders Qwen3 turns as
   `<|im_start|>role\n...<|im_end|>\n`.
2. Replaces tokenizer EOS with `<|im_end|>` as LLaMA-Factory does.
3. Adds an empty `<think>\n\n</think>\n\n` block when an assistant response has
   neither thinking delimiter.
4. Masks every system/user/prompt token with `-100`.
5. Applies LLaMA-Factory's source/target-aware truncation.
6. Reduces the packing capacity from 16,384 to 16,383, greedily packs within
   preprocessing batches of 1,000, then pads back to exactly 16,384.
7. Uses an all-ones attention mask and restarts position IDs for every packed
   conversation, matching the original default `neat_packing: false`.

`compare_preprocessing.py` can run the local processor and the reference
LLaMA-Factory processor on the same raw rows and compare all four tensor fields.

The repository's generated Parquet contains `role`/`content` messages.
`configs/sft/dataset_info.json` now declares those tags explicitly; without
that correction, current LLaMA-Factory defaults to `from`/`value`.

## 1. Environment on a fresh instance

Use an Ubuntu 24.04 Neuron PyTorch DLAMI. The AWS DLAMI already supplies the
driver and system tools; do not clone `torch-neuronx` into the working
directory.

The pinned stack supports Python 3.10 and 3.11. The setup wrapper automatically
selects either version; Python 3.11 is not required.

The setup wrapper creates an isolated environment with the pinned upstream
Optimum Neuron stack:

```bash
bash "$HOME/Lightning-OPD/trainium/optimum_neuron/setup_instance.sh"
source "$HOME/venvs/lightning-opd-sft/bin/activate"
```

The wrapper clears `PYTHONPATH` and disables user-site packages so an old
`$HOME/torch-neuronx` checkout cannot shadow the installed Neuron package.

The requirements pin the audited Optimum Neuron Qwen3 implementation and its
compatible Torch NeuronX/compiler versions. Verify the instance and imports:

```bash
neuron-ls

python - <<'PY'
import torch
import torch_neuronx
import optimum.neuron
from optimum.neuron import NeuronTrainer
from optimum.neuron.models.training import NeuronModelForCausalLM

print("torch:", torch.__version__)
print("torch-neuronx:", torch_neuronx.__version__)
print("torch-neuronx path:", torch_neuronx.__file__)
print("optimum-neuron:", optimum.neuron.__version__)
print("full trainer:", NeuronTrainer)
print("Qwen training model:", NeuronModelForCausalLM)
PY
```

The `torch_neuronx` path must point into this venv, not a source checkout such
as `$HOME/torch-neuronx`.

## 2. Pre-tokenize the production dataset once

Adjust the raw Parquet path to the location copied from the generation
instance:

```bash
cd "$HOME/Lightning-OPD"

python trainium/optimum_neuron/prepare_sft_dataset.py \
  --dataset-path /data/lightning-opd/data/sft/openthoughts3_300k_qwen3-8b.parquet \
  --output-dir /data/lightning-opd/data/sft/qwen3-4b-lf-packed-16384 \
  --model-id Qwen/Qwen3-4B-Base \
  --cutoff-len 16384 \
  --expected-rows 300000 \
  --preprocessing-num-workers 16 \
  --preprocessing-batch-size 1000 \
  --overwrite
```

This is the equivalent of the original `overwrite_cache: true`. It writes
`reproduction_metadata.json`; training refuses an unrecognized or wrong-length
tokenized dataset.

## 3. Smoke test

The smoke wrapper makes a separately packed 1,024-token dataset from the first
1,000 rows and runs two full-model optimizer steps on one TP8 group:

```bash
cd "$HOME/Lightning-OPD"
SFT_DATA=/data/lightning-opd/data/sft/openthoughts3_300k_qwen3-8b.parquet \
bash trainium/optimum_neuron/run_smoke_test.sh
```

Startup must print `FULL MODEL` and the full trainable parameter count.
Success ends with `SMOKE_TEST_OK`.

## 4. Precompile production shapes

Compilation uses the exact TP8, BF16, batch-1, and 16,384-token production
shapes:

```bash
TOKENIZED_DATA=/data/lightning-opd/data/sft/qwen3-4b-lf-packed-16384 \
SEQUENCE_LENGTH=16384 \
OUTPUT_DIR=/data/lightning-opd/runs/qwen3-4b-full-sft-compile \
neuron_parallel_compile \
  bash trainium/optimum_neuron/run_sft_trn2_48xlarge.sh \
  2>&1 | tee /data/lightning-opd/runs/qwen3-4b-full-sft-compile.log
```

## 5. Train

```bash
TOKENIZED_DATA=/data/lightning-opd/data/sft/qwen3-4b-lf-packed-16384 \
SEQUENCE_LENGTH=16384 \
OUTPUT_DIR=/data/lightning-opd/runs/qwen3-4b-full-sft \
bash trainium/optimum_neuron/run_sft_trn2_48xlarge.sh \
  2>&1 | tee /data/lightning-opd/runs/qwen3-4b-full-sft.log
```

Resume with the same output directory and an explicit checkpoint:

```bash
TOKENIZED_DATA=/data/lightning-opd/data/sft/qwen3-4b-lf-packed-16384 \
SEQUENCE_LENGTH=16384 \
OUTPUT_DIR=/data/lightning-opd/runs/qwen3-4b-full-sft \
RESUME_FROM_CHECKPOINT=/data/lightning-opd/runs/qwen3-4b-full-sft/checkpoint-100 \
bash trainium/optimum_neuron/run_sft_trn2_48xlarge.sh
```

## 6. Consolidate the full model

Optimum Neuron writes tensor-parallel full-model shards. Consolidate them to a
normal Hugging Face safetensors checkpoint:

```bash
optimum-cli neuron consolidate \
  /data/lightning-opd/runs/qwen3-4b-full-sft \
  /data/lightning-opd/runs/qwen3-4b-full-sft-hf

cp /data/lightning-opd/runs/qwen3-4b-full-sft/*.json \
  /data/lightning-opd/runs/qwen3-4b-full-sft-hf/
cp /data/lightning-opd/runs/qwen3-4b-full-sft/tokenizer* \
  /data/lightning-opd/runs/qwen3-4b-full-sft-hf/
```

There is no adapter merge step.

## Optional direct preprocessing comparison

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
git -C /tmp/LLaMA-Factory checkout \
  9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d

python trainium/optimum_neuron/compare_preprocessing.py \
  --llamafactory-repo /tmp/LLaMA-Factory \
  --dataset-path /data/lightning-opd/data/sft/openthoughts3_300k_qwen3-8b.parquet \
  --model-id Qwen/Qwen3-4B-Base \
  --cutoff-len 16384 \
  --num-samples 1000
```

Success prints a byte-for-byte match for `input_ids`, `attention_mask`,
`position_ids`, and `labels`.
