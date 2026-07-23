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
per chip. Compare `generation_seconds`, `generation_duration`, and
`generated_tokens_per_second` in the two `benchmark.json` files. Generation
wall time starts immediately before the worker processes launch and ends after
all workers finish, so TP=2 measures both replicas concurrently rather than
adding their individual runtimes. `total_run_seconds` also includes environment
setup, downloads, compilation, and output merging. A run is suitable for
scaling only if it compiles, completes all eight
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

## Two trn2.48xlarge instances

The two-instance entrypoint uses the validated TP=4, `max_num_seqs=16`
topology. Prepare the global prompt set on a helper Trainium instance, then
split it into two physical 150K-row files:

```bash
ALLOW_INSTANCE_MISMATCH=1 PREPARE_ONLY=1 \
  WORK_ROOT=/data/lightning-opd-prep \
  bash trainium/nxd/run_sft_curation_trn2_48xlarge.sh

python3 data_curation/split_jsonl_shards.py \
  --input /data/lightning-opd-prep/prompts/openthoughts3_300000.jsonl \
  --output-dir /data/lightning-opd-prep/prompts/shards \
  --num-shards 2 \
  --expected-rows 300000 \
  --prefix openthoughts3_300000
```

Upload the shard directory to S3 manually:

```bash
aws s3 cp /data/lightning-opd-prep/prompts/shards/ \
  s3://qwen3/lightning-opd/qwen3-8b-300k/prompts/ --recursive
```

Before launching, manually download the matching shard into
`WORK_ROOT/prompts` on each large instance. Then run:

```bash
INSTANCE_RANK=0 WORK_ROOT=/data/lightning-opd \
  bash trainium/nxd/run_sft_curation_2x_trn2_48xlarge.sh

INSTANCE_RANK=1 WORK_ROOT=/data/lightning-opd \
  bash trainium/nxd/run_sft_curation_2x_trn2_48xlarge.sh
```

No distributed rendezvous, EFA setup, or cross-instance process group is
needed. Each machine divides its 150K-row input across 16 local data ranks.
The per-instance results are:

```text
<WORK_ROOT>/sft_data/openthoughts3_300000_qwen3-8b_node00000-of-00002.parquet
<WORK_ROOT>/sft_data/openthoughts3_300000_qwen3-8b_node00001-of-00002.parquet
```

Upload both node results to S3 manually. On node 0, manually download the
node-1 result, then stream-merge and validate:

```bash
python data_curation/merge_parquet_shards.py \
  --inputs \
    /path/to/openthoughts3_300000_qwen3-8b_node00000-of-00002.parquet \
    /path/to/openthoughts3_300000_qwen3-8b_node00001-of-00002.parquet \
  --output /path/to/openthoughts3_300000_qwen3-8b.parquet \
  --expected-rows 300000
```

## Documented compatibility path

- [vLLM User Guide for NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html)
- [AWS Neuron logical NeuronCore configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html)
