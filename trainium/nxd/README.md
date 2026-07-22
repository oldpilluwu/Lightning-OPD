# Trainium NxD data curation

This directory contains the paper-faithful SFT data-curation entrypoint for
the Lightning OPD 4B student path. It uses the NxD Inference-backed vLLM
integration because that is the supported Neuron vLLM implementation for the
text-only `Qwen/Qwen3-8B` architecture.

## Target

- EC2 `trn2.48xlarge`
- Ubuntu 24.04 AWS Neuron DLAMI
- PyTorch 2.9 NxD Inference virtual environment
- 16 Trainium2 chips, exposed as 64 logical NeuronCores with LNC=2

The launcher uses 16 independent Qwen3-8B replicas. Each replica is tensor
parallel across four logical NeuronCores, so all 16 chips curate separate
dataset shards concurrently.

## Run

From the repository root:

```bash
bash trainium/nxd/run_sft_curation_trn2_48xlarge.sh
```

For a full run, place caches and generated data on a volume with several
hundred GiB free. Local NVMe is preferred for compilation and temporary
Arrow shards:

```bash
WORK_ROOT=/path/on/local/nvme/lightning-opd \
  bash trainium/nxd/run_sft_curation_trn2_48xlarge.sh
```

Set `HF_TOKEN` in the environment if your Hugging Face configuration requires
one. Set `INFER_VENV` only when the DLAMI's NxD environment is not in one of
the standard locations detected by `setup_env.sh`.

## One-chip smoke test

Use `trn2.3xlarge` to compile Qwen3-8B and generate eight real OpenThoughts3
responses before starting the full run. The smoke test fetches only the rows it
needs through the Hugging Face Dataset Viewer, downloads the model to a reusable local path, retains the production
18,432-token compiled sequence length, and writes a JSON throughput report.

Start with the memory-safe TP=4 layout:

```bash
bash trainium/nxd/run_sft_smoke_trn2_3xlarge.sh
```

Then test two TP=2 replicas on the same chip:

```bash
TP_SIZE=2 bash trainium/nxd/run_sft_smoke_trn2_3xlarge.sh
```

The TP=4 run uses one replica with eight sequence slots. The TP=2 run uses two
replicas with four slots each, so both configurations expose eight total slots
per chip. Compare `generated_tokens_per_second` in the two `benchmark.json`
files. A run is suitable for scaling only if it compiles, completes all eight
long-context prompts without a Neuron out-of-memory/runtime error, and produces
a valid eight-row parquet. Each successful smoke run prints the matching
`trn2.48xlarge` command.

The default 16K response cap deliberately matches production. For a quick
installation-only check, it can be reduced, but that result does not establish
long-generation stability or representative throughput:

```bash
MAX_TOKENS=256 SMOKE_SAMPLES=2 bash trainium/nxd/run_sft_smoke_trn2_3xlarge.sh
```

The end-to-end launcher:

1. Activates and verifies the NxD inference environment.
2. Installs `vllm-neuron` 0.5.0 and data dependencies when needed.
3. Downloads and deterministically samples 300K OpenThoughts3 prompts.
4. Downloads Qwen3-8B to a local path, as required for its tied embeddings.
5. Compiles the TP=4 model once and shares the compiled artifacts.
6. Runs 16 resumable vLLM workers with paper sampling settings.
7. Merges and validates the final 300K-row SFT parquet.

Default output:

```text
data/trainium/nxd/sft_data/openthoughts3_300000_qwen3-8b.parquet
```

Each completed rank has an `_SUCCESS` marker. If the instance or process is
interrupted, rerun the same command; completed ranks are skipped and partial
ranks resume from their batch checkpoints.

## Documented compatibility path

- [vLLM User Guide for NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html)
- [AWS Neuron logical NeuronCore configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html)
