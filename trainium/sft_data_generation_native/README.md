# Native PyTorch SFT data curation

This is the Trainium-local native implementation. It intentionally lives next
to `trainium/sft_data_generation/` so the repository's original
`data_curation/` code can stay as the format/merge layer.

This directory replaces the SFT teacher-generation use of vLLM/NxD with plain
Hugging Face Transformers on the private Beta-3 TorchNeuron backend. It is an
experimental/reference path for Lightning-OPD, not an OpenAI-compatible server.

## Design

- Qwen3-8B BF16 model forward on `torch.device("neuron")`.
- `attn_implementation="eager"` so left-padding masks are honored.
- CPU FP32 top-p sampling; only the small logits sampling operation leaves the device.
- Fixed prompt bucket, fixed full attention mask, and `StaticCache` for static shapes.
- Phase 1 is eager Neuron correctness. Phase 2 uses
  `torch.compile(backend="neuron", dynamic=False)`. CPU FP32 is the validation
  reference and optional failure fallback, never reported as Neuron success.
- Multiple cores are data-parallel independent replicas. Tensor parallelism would
  require NxD and is intentionally unavailable in this pure-native path.

The downstream format is unchanged: Arrow shards contain `messages` and an integer
generated-token count in `tokens`; `data_curation/merge.py` produces the final parquet.

## Beta-3 setup

Use the matching private Beta-3 DLC/runtime/driver described by the supplied Native
PyTorch User Guide. If you already copied the DLC `/workspace` artifacts to
`$HOME/workspace`, this creates and validates `$HOME/workspace/native_venv`:

```bash
NATIVE_VENV=$HOME/workspace/native_venv \
  bash trainium/sft_data_generation_native/setup_env.sh
```

If the artifacts are not present yet, pass the private DLC URI from the guide or
your AWS account team:

```bash
BETA_IMAGE_URI=<private-beta-dlc-uri> \
  bash trainium/sft_data_generation_native/setup_env.sh
```

The setup script does not install packages by default. If the environment is missing
the ordinary Python data dependencies and you approve installing them, use
`INSTALL_DEPS=1`.

## Smoke and full runs

The faithful smoke keeps the paper's source/model/sampling/generation configuration
(OpenThoughts3, seed 42, 64 prompts, Qwen3-8B, 16,384 max new tokens, temperature 0.7,
top-p 0.9, batch size 1), runs CPU-vs-eager-vs-compiled validation, and reduces only dataset size:

```bash
SMOKE=1 bash trainium/sft_data_generation_native/generate_sft_data.sh
```

For a quick engineering check that intentionally shortens the workload:

```bash
FAST_SMOKE=1 VALIDATE=0 bash trainium/sft_data_generation_native/generate_sft_data.sh
```

The full paper-scale curation is:

```bash
bash trainium/sft_data_generation_native/generate_sft_data.sh
```

Useful experiment overrides include `MODE=eager`, `MODE=compile`, `NUM_CORES`,
`BATCH_SIZE`, `PREFILL_BUCKET`, `MAX_TOKENS`, and `VALIDATE=0`. Start on one core;
increase `NUM_CORES` only after accounting for one Qwen3-8B replica per worker.
