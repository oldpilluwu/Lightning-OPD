# Qwen3.5 2B <- 9B Experiment Scripts

These scripts run a drift-observation Lightning-OPD experiment with:

- Student base: `Qwen/Qwen3.5-2B-Base`
- Teacher: `Qwen/Qwen3.5-9B`
- Default drift-observation sizes: `5000` SFT prompts and `2000` OPD prompts

## Remote Setup

On a fresh Linux GPU server:

```bash
git clone <your-repo-url> Lightning-OPD
cd Lightning-OPD
bash scripts/qwen35_2b_9b/setup_remote.sh
```

The setup script creates three conda envs:

- `qwen35-curation` for vLLM data generation and eval
- `qwen35-sft` for LlamaFactory SFT
- `qwen35-train` for SGLang, Ray, FSDP, logprob precompute, and Lightning-OPD

## One Command

Run the full pipeline, export the final HF model, inspect drift logs, and run a small DAPO-math sanity eval:

```bash
bash scripts/qwen35_2b_9b/run_all.sh
```

Outputs go under:

```text
logs/qwen35_2b_9b/<RUN_ID>/
```

Key files:

- `drift/metrics.csv`
- `drift/summary.txt`
- `eval_sft.json`
- `eval_final.json`
- `tensorboard/`

## Manual Steps

Run from the repo root if you want to execute stages one by one.

```bash
bash scripts/qwen35_2b_9b/00_prepare_prompts.sh
bash scripts/qwen35_2b_9b/01_generate_sft_data.sh
bash scripts/qwen35_2b_9b/02_run_sft.sh
bash scripts/qwen35_2b_9b/03_collect_rollouts.sh
bash scripts/qwen35_2b_9b/05_precompute_teacher_logprobs.sh
bash scripts/qwen35_2b_9b/06_train_lightning_opd_fsdp.sh
python scripts/qwen35_2b_9b/inspect_drift.py --log-dir logs/qwen35_2b_9b/<RUN_ID> --output-dir logs/qwen35_2b_9b/<RUN_ID>/drift
bash scripts/qwen35_2b_9b/07_export_latest_hf.sh
bash scripts/qwen35_2b_9b/08_eval_math.sh
```

Useful overrides:

```bash
NUM_GPUS=1 TP_SIZE=1 BATCH_SIZE=2 MAX_TOKENS=2048 bash scripts/qwen35_2b_9b/01_generate_sft_data.sh
SFT_MAX_STEPS=1000 SFT_CUTOFF_LEN=8192 bash scripts/qwen35_2b_9b/02_run_sft.sh
START_TEACHER=0 TEACHER_URL=http://127.0.0.1:13141/generate bash scripts/qwen35_2b_9b/05_precompute_teacher_logprobs.sh
NUM_ROLLOUT=300 ROLLOUT_BATCH_SIZE=8 GLOBAL_BATCH_SIZE=8 bash scripts/qwen35_2b_9b/06_train_lightning_opd_fsdp.sh
```

## Inspect While Running

```bash
tail -f logs/qwen35_2b_9b/<RUN_ID>/06_train_lightning_opd_fsdp.log
tensorboard --logdir logs/qwen35_2b_9b/<RUN_ID>/tensorboard --host 0.0.0.0 --port 6006
python scripts/qwen35_2b_9b/inspect_drift.py --log-dir logs/qwen35_2b_9b/<RUN_ID> --output-dir logs/qwen35_2b_9b/<RUN_ID>/drift
```

Watch these drift signals:

- `rollout/log_probs`
- `rollout/ref_log_probs`
- `rollout/advantages`
- `train/ppo_kl`
- `train/kl_loss`
- `train/entropy`

Suggested machine split:

- Free RTX 5090: prompt generation, SFT data generation, small rollouts.
- RTX A6000: Qwen3.5-9B teacher serving/logprob precompute if the 5090 stack is unstable.
- A100 80GB: SFT/Lightning-OPD training once the pilot data path works.

Qwen3.5 requires recent `transformers`, `vllm`, and `sglang`. If serving fails on an unknown argument, set `SGLANG_TP_FLAG=--tp-size` or remove optional args through `SGLANG_EXTRA_ARGS`.
