# Qwen3-4B full SFT on one trn2.48xlarge

This is full-parameter training. There is no PEFT configuration, no adapter,
and no LoRA merge step. The backend is the DLAMI's Neuron PyTorch 2.9/XLA
stack, NeuronX Distributed, and the audited Optimum Neuron custom Qwen3 model.

This exact combination is a compatibility adaptation rather than an upstream
Optimum Neuron release: Optimum has not published a PyTorch 2.9/Python 3.12
package. AWS documents that 2.8-to-2.9 training scripts require no changes, and
the Qwen3 model's required NxD APIs remain present in the PyTorch 2.9 stack.
The topology smoke test is therefore a required gate before production.

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

Use the Ubuntu 24.04 Neuron PyTorch DLAMI and its preinstalled
`aws_neuronx_venv_pytorch_2_9` environment. Python 3.12 is supported by the
Neuron PyTorch 2.9 binary stack; Python 3.11 is not required. The overlaid
Optimum source revision itself declares Python `<3.12`, so using it with the
DLAMI's Python 3.12 remains an explicit compatibility override that the smoke
test must validate on the instance.

The audited Optimum Neuron revision contains the custom Qwen3 training model,
but its package metadata pins the older Neuron PyTorch 2.8 binaries. To honor
the PyTorch 2.9 requirement, the setup wrapper:

1. Uses the preinstalled PyTorch 2.9 venv without creating another venv.
2. Verifies `torch` and `torch-neuronx` are both version 2.9.
3. Installs pinned framework-neutral Python dependencies and verifies that no
   AWS Neuron binary package version changed.
4. Checks out the audited Optimum Neuron source without installing its pinned
   Neuron binary dependencies.
5. Makes the launcher load that Qwen3 Python source on top of the DLAMI's
   PyTorch 2.9, Torch/XLA, compiler, and NeuronX Distributed packages.

```bash
source "$HOME/aws_neuronx_venv_pytorch_2_9/bin/activate"
unset PYTHONPATH

bash "$HOME/Lightning-OPD/trainium/optimum_neuron/setup_instance.sh"
```

If the venv is under `/opt`, activate
`/opt/aws_neuronx_venv_pytorch_2_9/bin/activate` instead. The wrapper detects
either location. It clears `PYTHONPATH` and disables user-site packages so an
old `$HOME/torch-neuronx` checkout cannot shadow the installed Neuron package.

Verify the instance and imports:

```bash
neuron-ls
export OPTIMUM_NEURON_SRC="$HOME/optimum-neuron-qwen3-pt29"
export PYTHONPATH="$OPTIMUM_NEURON_SRC"

python - <<'PY'
from importlib.metadata import version
import torch
import torch_neuronx
import torch_xla
import torch_xla.runtime as xr
import optimum.neuron
from optimum.neuron import NeuronTrainer
from optimum.neuron.models.training import NeuronModelForCausalLM

print("torch:", torch.__version__)
print("torch-neuronx:", torch_neuronx.__version__)
print("torch-neuronx path:", torch_neuronx.__file__)
print("torch-xla:", version("torch-xla"))
print("XLA device:", torch_xla.device())
print("runtime device count:", xr.global_runtime_device_count())
print("optimum-neuron:", optimum.neuron.__version__)
print("optimum-neuron source:", optimum.neuron.__file__)
print("full trainer:", NeuronTrainer)
print("Qwen training model:", NeuronModelForCausalLM)
PY
```

The `torch_neuronx` path must point into
`aws_neuronx_venv_pytorch_2_9`, not a source checkout such as
`$HOME/torch-neuronx`. The setup accepts the DLAMI-provided NxD 0.18 and 0.19
lines used by AWS PyTorch 2.9 releases—including `0.19.28492`—and refuses to
install or upgrade a missing Neuron binary package. The runtime device count
should be 64 on a `trn2.48xlarge` using the default LNC=2 configuration.

The setup keeps NumPy on the 2.x line required by the current DLAMI SciPy
build and force-reinstalls only the pinned pure-Python Transformers package.
This repairs stale or mixed Transformers files without reinstalling Torch,
Torch-XLA, Torch-NeuronX, NxD, libneuronxla, or neuronx-cc.

## Dataset layout used by this project

Data generation and training both ran on one `trn2.48xlarge`. The generated
dataset is already in the sibling `lightning-opd-prep` directory; do not
download, regenerate, merge, or copy it:

```text
$HOME/
  Lightning-OPD/
  lightning-opd-prep/
    sft_data/
      openthoughts3_300000_qwen3-8b.parquet
```

Set these once after activating the environment:

```bash
export REPO_DIR="$HOME/Lightning-OPD"
export PREP_DIR="$HOME/lightning-opd-prep"
export SFT_DATA="$PREP_DIR/sft_data/openthoughts3_300000_qwen3-8b.parquet"
export TOKENIZED_DATA="$PREP_DIR/sft_data/qwen3-4b-lf-packed-16384"
export OUTPUT_ROOT="$PREP_DIR/training_runs"

test -f "$SFT_DATA"
```

## 2. Pre-tokenize the production dataset once

Use the generated Parquet in the sibling preparation directory:

```bash
cd "$REPO_DIR"

python trainium/optimum_neuron/prepare_sft_dataset.py \
  --dataset-path "$SFT_DATA" \
  --output-dir "$TOKENIZED_DATA" \
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
1,000 rows and runs two full-model optimizer steps on 16 workers (two TP8
groups). Sixteen is the smallest Trn2-supported XLA worker count divisible by
TP8. It automatically resolves the sibling path shown above, so no data
argument is needed:

```bash
cd "$REPO_DIR"
bash trainium/optimum_neuron/run_smoke_test.sh
```

Startup must print `FULL MODEL` and the full trainable parameter count.
Success ends with `SMOKE_TEST_OK`. If the preparation directory is elsewhere,
pass `PREP_DIR=/other/path` or `SFT_DATA=/other/file.parquet`.

## 4. Precompile production shapes

Compilation uses the exact TP8, BF16, batch-1, and 16,384-token production
shapes:

```bash
mkdir -p "$OUTPUT_ROOT"

TOKENIZED_DATA="$TOKENIZED_DATA" \
SEQUENCE_LENGTH=16384 \
OUTPUT_DIR="$OUTPUT_ROOT/qwen3-4b-full-sft-compile" \
neuron_parallel_compile \
  bash trainium/optimum_neuron/run_sft_trn2_48xlarge.sh \
  2>&1 | tee "$OUTPUT_ROOT/qwen3-4b-full-sft-compile.log"
```

## 5. Train

```bash
TOKENIZED_DATA="$TOKENIZED_DATA" \
SEQUENCE_LENGTH=16384 \
OUTPUT_DIR="$OUTPUT_ROOT/qwen3-4b-full-sft" \
bash trainium/optimum_neuron/run_sft_trn2_48xlarge.sh \
  2>&1 | tee "$OUTPUT_ROOT/qwen3-4b-full-sft.log"
```

Resume with the same output directory and an explicit checkpoint:

```bash
TOKENIZED_DATA="$TOKENIZED_DATA" \
SEQUENCE_LENGTH=16384 \
OUTPUT_DIR="$OUTPUT_ROOT/qwen3-4b-full-sft" \
RESUME_FROM_CHECKPOINT="$OUTPUT_ROOT/qwen3-4b-full-sft/checkpoint-100" \
bash trainium/optimum_neuron/run_sft_trn2_48xlarge.sh
```

## 6. Consolidate the full model

Optimum Neuron writes tensor-parallel full-model shards. Consolidate them to a
normal Hugging Face safetensors checkpoint:

```bash
optimum-cli neuron consolidate \
  "$OUTPUT_ROOT/qwen3-4b-full-sft" \
  "$OUTPUT_ROOT/qwen3-4b-full-sft-hf"

cp "$OUTPUT_ROOT/qwen3-4b-full-sft/"*.json \
  "$OUTPUT_ROOT/qwen3-4b-full-sft-hf/"
cp "$OUTPUT_ROOT/qwen3-4b-full-sft/"tokenizer* \
  "$OUTPUT_ROOT/qwen3-4b-full-sft-hf/"
```

There is no adapter merge step.

## Optional direct preprocessing comparison

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git /tmp/LLaMA-Factory
git -C /tmp/LLaMA-Factory checkout \
  9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d

python trainium/optimum_neuron/compare_preprocessing.py \
  --llamafactory-repo /tmp/LLaMA-Factory \
  --dataset-path "$SFT_DATA" \
  --model-id Qwen/Qwen3-4B-Base \
  --cutoff-len 16384 \
  --num-samples 1000
```

Success prints a byte-for-byte match for `input_ids`, `attention_mask`,
`position_ids`, and `labels`.
