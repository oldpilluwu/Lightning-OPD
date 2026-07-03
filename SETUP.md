# Lightning OPD on AWS Trainium — Setup Guide

This guide takes you from a fresh AWS account to a trained Lightning-OPD model
(Qwen3-4B or Qwen3-8B scale) on AWS Trainium, using the scripts in
[`trainium/`](trainium/).

> **Read this first — what this port is and isn't.**
> The original pipeline runs on CUDA (vLLM, LlamaFactory, sglang,
> Megatron/slime). None of those run on Trainium, so the `trainium/` scripts
> re-implement each step on the AWS Neuron stack with **identical data
> formats, hyperparameters, and loss math**:
>
> | Step | Original (CUDA) | Trainium port |
> |---|---|---|
> | 0. Prompts | `scripts/prepare_sft_prompts.py` (CPU) | **unchanged** |
> | 1. SFT data gen | vLLM (CUDA) | vLLM **Neuron plugin** — `trainium/pipeline_neuron.py` |
> | 2. SFT | LlamaFactory + DeepSpeed | **optimum-neuron** `NeuronTrainer` — `trainium/step2_sft_train_neuron.py` (same template/masking/packing/LR schedule) |
> | 3. Rollouts | vLLM (CUDA) | same as step 1, with the SFT model |
> | 4. Teacher logprobs | sglang server + HTTP | offline vLLM **prompt-logprob scoring** — `trainium/step4_teacher_logprobs_neuron.py` (same quantity, same parquet schema) |
> | 5. Lightning OPD | slime (Megatron+Ray) | custom `NeuronTrainer` loss — `trainium/step5_lightning_opd_train_neuron.py` (advantage = teacher logprob − student logprob, lr 2e-6, batch 256, identical) |
> | 6. Ckpt convert | Megatron→HF script | `optimum-cli neuron consolidate` (built into the pipeline) |
>
> These scripts were written against **Neuron SDK ≥ 2.24** (Qwen3 support in
> NxD Inference) and **optimum-neuron ≥ 0.3.0** (Qwen3 training support) and
> have **not been validated on real Trainium hardware** — run the smoke test
> (§7) before committing to the full run. The two known risk points are
> called out in §10 (Troubleshooting).

---

## 1. Choose instance type and region

Everything runs on **one `trn1.32xlarge`** (16 Trainium1 chips = 32
NeuronCores, 512 GB accelerator memory, 128 vCPUs, ~$21.50/hr on-demand).

- Regions with Trn1: **us-east-1 (N. Virginia)**, **us-east-2 (Ohio)**,
  **us-west-2 (Oregon)**. Pick one and stay in it.
- `trn2.48xlarge` (Trainium2, ~4x faster) also works with the same scripts
  if you have access, but it is more expensive and quota is harder to get.
- Do **not** try `trn1.2xlarge` — a single chip (2 cores × 16 GB) cannot hold
  the models with optimizer state.

Rough budget (full-scale reproduction, on-demand pricing — the dominant cost
is step 1 generating 300k × up-to-16k-token responses):

| Scale | Step 1 (data gen) | Step 2 (SFT) | Steps 3–4 | Step 5 (OPD) | Total (very rough) |
|---|---|---|---|---|---|
| 4B (teacher 8B) | ~40–60 h | ~30–45 h | ~8–12 h | ~3–5 h | **~80–120 h ≈ $1.7k–2.6k** |
| 8B (teacher 32B) | ~90–140 h | ~50–70 h | ~15–25 h | ~5–8 h | **~160–240 h ≈ $3.5k–5.2k** |

The paper's numbers (20/30 GPU-hours) are for the OPD stage only, on H100s.
If you only want to validate the pipeline, run the smoke test first (§7,
a few hours, <$100).

## 2. Request quota (AWS Console)

New accounts have **zero** Trn1 quota. Do this first — approval can take a
few hours to a couple of days.

1. Sign in to the AWS Console, set your region (top-right) to e.g. **us-east-2**.
2. Search bar → **Service Quotas** → **AWS services** → **Amazon Elastic
   Compute Cloud (Amazon EC2)**.
3. Search the quota list for **"Running On-Demand Trn instances"**.
4. **Request increase at account level** → enter **128** (vCPUs; a
   trn1.32xlarge uses all 128). Submit.
5. Watch status under *Quota request history*. If it stalls, open a Support
   case (Support Center → Create case → Service limit increase) and mention
   you're running an ML training job.

## 3. Create a key pair

1. Console → **EC2** → left sidebar **Network & Security → Key Pairs** →
   **Create key pair**.
2. Name `trainium-key`, type **ED25519**, format **.pem** → Create.
3. The `.pem` downloads automatically. On your machine (PowerShell):
   ```powershell
   # Windows: restrict permissions so ssh accepts the key
   icacls $HOME\Downloads\trainium-key.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ```

## 4. Launch the instance (AWS Console)

1. Console → **EC2** → **Launch instance**.
2. **Name**: `lightning-opd-trn1`.
3. **Application and OS Images (AMI)** → *Browse more AMIs* → search
   **"Deep Learning AMI Neuron"** → pick
   **Deep Learning AMI Neuron (Ubuntu 22.04)** (the multi-framework Neuron
   DLAMI; it ships the Neuron driver, runtime, and the PyTorch/NxD-Inference
   virtualenvs the scripts expect).
4. **Instance type**: `trn1.32xlarge`.
5. **Key pair**: `trainium-key`.
6. **Network settings** → Edit:
   - Keep the default VPC/subnet, **Auto-assign public IP: Enable**.
   - Security group: *Create security group*, allow **SSH (22)** with source
     **My IP** only.
7. **Configure storage**: set the root volume to **2000 GiB, gp3** (the
   300k-sample SFT dataset + checkpoints + Neuron compile cache need room;
   1000 GiB is enough for the smoke test). Optionally raise gp3 throughput
   to 500 MB/s.
8. **Launch instance**. Wait until *Instance state = Running* and note the
   **Public IPv4 address** from the instance details page.

> 💡 Cost control: this instance bills ~$21.50/hr while running. **Stop** it
> (EC2 → Instance state → Stop) whenever you're not training — the EBS
> volume (and all your data) persists across stop/start; only the public IP
> changes. The resumable pipeline picks up where it left off.

## 5. Connect and set up

```bash
ssh -i ~/Downloads/trainium-key.pem ubuntu@<PUBLIC_IP>
```

Verify the accelerators — you should see 16 devices / 32 NeuronCores:

```bash
neuron-ls
```

Get the code onto the instance. Either push this repo (with the `trainium/`
directory committed) to your GitHub and clone it:

```bash
git clone https://github.com/<your-user>/Lightning-OPD.git
cd Lightning-OPD
```

…or copy it from your machine (PowerShell):

```powershell
scp -i $HOME\Downloads\trainium-key.pem -r C:\Users\fawwa\projects\Lightning-OPD ubuntu@<PUBLIC_IP>:~/
```

Then install the two Python environments (inference = vLLM-on-Neuron;
training = optimum-neuron):

```bash
cd ~/Lightning-OPD
bash trainium/setup_env.sh
```

Check the venv names it prints; if your DLAMI uses a different PyTorch
version (e.g. `aws_neuronx_venv_pytorch_2_8`), pass them explicitly:

```bash
INFER_VENV=/opt/<nxd_inference_venv> TRAIN_VENV=/opt/<pytorch_venv> bash trainium/setup_env.sh
```

Optional but recommended:

```bash
# HF cache on the big volume + faster downloads
echo 'export HF_HOME=/home/ubuntu/hf_cache' >> ~/.bashrc
# Weights & Biases logging for the training stages
echo 'export WANDB_API_KEY=<your-key>' >> ~/.bashrc
source ~/.bashrc
```

## 6. Smoke-test the two risk points (15–30 min, do not skip)

**(a) Qwen3 generation on Neuron** — verifies the vLLM Neuron plugin +
Qwen3 support:

```bash
source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-0.6B", tensor_parallel_size=2, max_model_len=2048, max_num_seqs=4)
out = llm.chat([[{"role": "user", "content": "What is 2+2?"}]], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
PY
```

**(b) Prompt logprobs** (needed by step 4):

```bash
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-0.6B", tensor_parallel_size=2, max_model_len=2048, max_num_seqs=4)
out = llm.generate(["The capital of France is Paris."], SamplingParams(max_tokens=1, prompt_logprobs=0))
print(out[0].prompt_logprobs[:5])
PY
```

If (b) prints a list of per-token logprob dicts, step 4 will work. If your
Neuron vLLM build rejects `prompt_logprobs`, see §10.

## 7. Run the pipeline

Always inside `tmux` so an SSH drop doesn't kill a multi-day run:

```bash
tmux new -s opd
cd ~/Lightning-OPD
```

**Recommended first: end-to-end smoke run** (5k SFT samples, 2k rollouts,
20 OPD steps — validates every stage cheaply):

```bash
SCALE=4b SMOKE=1 bash trainium/run_pipeline.sh
```

**Full 4B reproduction** (student Qwen3-4B-Base, teacher Qwen3-8B):

```bash
SCALE=4b bash trainium/run_pipeline.sh
```

**Full 8B reproduction** (student Qwen3-8B-Base, teacher Qwen3-32B):

```bash
SCALE=8b bash trainium/run_pipeline.sh
```

The script runs step 0 → 6 in order, switching venvs automatically. Each
completed stage writes a marker under `data/.pipeline_state/`, and the
long stages (generation, teacher scoring) checkpoint internally — so after
any crash, instance stop, or Ctrl-C, **just re-run the same command** and it
resumes. To force a stage to re-run, delete its
`data/.pipeline_state/<scale>/<stage>.done` marker.

Detach from tmux with `Ctrl-b d`; reattach later with `tmux attach -t opd`.
Monitor progress in a second SSH session with `neuron-top`.

Individual steps can also be run directly — each `trainium/step*.sh` script
documents its required env vars in its header, mirroring the original
`scripts/*.sh` exactly.

Useful knobs (env vars for `run_pipeline.sh`):

| Var | Default | Meaning |
|---|---|---|
| `SFT_SAMPLES` | 300000 | SFT prompts sampled from OpenThoughts3-1.2M |
| `SFT_STEPS` | 3000 | SFT optimizer steps (paper value) |
| `OPD_STEPS` | 150 | Lightning OPD steps (README: ~150 converges; paper config caps at 3000) |
| `GEN_TP` | 8 | NeuronCores per vLLM worker in generation/scoring |
| `SFT_GBS` | 256 (4b) / 128 (8b) | SFT global batch (matches LlamaFactory configs) |
| `CUTOFF_LEN` | 16384 | SFT packed sequence length (lower to 8192 if OOM) |

## 8. Deciding when SFT is enough (OPD readiness probe)

The pipeline builds a **frozen probe set** before SFT starts
(`data/probe/opd_probe_<teacher>.parquet`): the teacher generates responses
on 64 held-out OPD prompts (DAPO-Math-17k) and scores its own tokens at
temperature 0 (same semantics as step 4). During SFT — and again during OPD —
the student is evaluated on this probe every `PROBE_EVERY` steps (default 5;
set `PROBE_EVERY=1` for literally every step at ~2–5% extra wall time).

Metrics land in `<checkpoint_dir>/opd_probe_log.jsonl`, stdout, and wandb
(`opd_probe/*`) if enabled:

| Metric | Meaning |
|---|---|
| `student_nll` | Student per-token NLL on teacher responses to OPD prompts — "how surprised is the student by what the teacher would say". |
| `teacher_nll` | Teacher NLL on its own tokens (constant; its sampling entropy). Floor for `student_nll`. |
| `fwd_kl` | `student_nll − teacher_nll` = Monte-Carlo estimate of per-token KL(teacher‖student) on the OPD domain. **The decision metric.** |
| `drift_mean` / `drift_abs` | Signed / absolute change in student token logprobs since the previous probe — how much the policy moved between probes. |
| `top1_match` | Fraction of teacher tokens that are the student's argmax. |

**Decision rule — move from SFT to OPD when:**

1. `fwd_kl` has **plateaued**: less than ~1–2% relative improvement over the
   last few hundred steps (compare checkpoints at `save_steps` boundaries), **and**
2. `drift_abs` is still clearly non-zero — i.e., SFT is still changing the
   policy, just not moving it closer to the teacher. Continuing past this
   point buys nothing on the OPD domain and risks drifting the student away
   from its own rollout distribution.

Also sanity-check `top1_match` (should have flattened) and the ordinary SFT
train loss (should be smooth). If `fwd_kl` is still falling steadily at step
3000, the paper's step count is binding you — extend `SFT_STEPS`.

Pick the SFT checkpoint at (or just after) the plateau point for steps 3–5,
not necessarily the last one: `checkpoints/<sft-dir>/checkpoint-<step>` can be
consolidated the same way as the final model.

Quick plot on the instance:

```bash
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("checkpoints/qwen3-4b-base-sft-qwen3-8b/opd_probe_log.jsonl")]
for r in rows[-30:]:
    bar = "#" * int(max(0, r["fwd_kl"]) * 40)
    print(f"step {r['step']:5d}  fwd_kl {r['fwd_kl']:.4f}  "
          f"drift_abs {r.get('drift_abs', float('nan')):.4f}  "
          f"top1 {r['top1_match']:.3f}  {bar}")
PY
```

The same probe keeps running during step 5, so you can watch OPD push
`fwd_kl` below the level SFT plateaued at — that drop is the Lightning-OPD
gain, visible before you run any benchmark. (Note: OPD optimizes *reverse*
KL on student rollouts, while `fwd_kl` measures forward KL on teacher
samples — related but not identical quantities, so expect improvement, not
monotone descent to zero.)

## 9. Get the final model + back up to S3

The pipeline ends with a consolidated HuggingFace-format model at:

```
checkpoints/qwen3-4b-lightning-opd-hf/    # or qwen3-8b-...
```

Back it up before terminating anything:

```bash
aws s3 mb s3://<your-bucket>-lightning-opd
aws s3 sync checkpoints/qwen3-4b-lightning-opd-hf s3://<your-bucket>-lightning-opd/qwen3-4b-lightning-opd-hf
```

(If `aws s3` complains about credentials: EC2 console → select instance →
Actions → Security → Modify IAM role → attach a role with S3 write access;
or run `aws configure` with an access key.)

Evaluation (AIME / HMMT / LiveCodeBench) is not part of this repo's pipeline;
the checkpoint loads with vLLM/transformers on any hardware.

## 10. Troubleshooting

- **First training step takes 30–60+ min** — that's the Neuron compiler
  tracing and compiling the graphs (once per shape). Subsequent steps are
  fast; the cache persists in `/var/tmp/neuron-compile-cache` across runs.
  You can pre-populate caches for step 2/5 with `neuron_parallel_compile`.
- **`prompt_logprobs` not supported by your Neuron vLLM build (step 4)**:
  upgrade to the newest Neuron SDK / vLLM-neuron release branch first. If
  it's still unsupported, the fallback is to compute teacher logprobs with
  plain forward passes (`torch.log_softmax` over teacher logits, gathering
  the actual next token) via `transformers` on `torch_xla` — open an issue or
  ask for `--backend xla` to be added to `step4_teacher_logprobs_neuron.py`;
  the math is 15 lines, the missing part is only teacher-size tensor
  parallelism on trn1.
- **optimum-neuron API drift (steps 2/5)**: the trainer scripts follow the
  optimum-neuron 0.3.x training API (`NeuronTrainingArguments(tensor_parallel_size=…)`,
  `NeuronModelForCausalLM.from_pretrained(model_id, training_args.trn_config)`).
  If imports fail on your version, check
  https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3
  and adjust the few construction lines — the datasets, loss, and
  hyperparameters don't need to change.
- **Device OOM in step 2**: lower `CUTOFF_LEN` to 8192 (halves activation
  memory; slightly changes packing but not the data), or raise `TP_SIZE`
  to 16.
- **Device OOM in step 5**: raise `TP_SIZE` (the default 8 is conservative
  for 4B; 8B may need 16 on trn1), or reduce `--max-seq-len` if your rollouts
  have short prompts.
- **HBM OOM or hangs in generation (steps 1/3)**: reduce `MAX_NUM_SEQS`
  (default 8) or `MAX_MODEL_LEN` (default 18432; 4096-token rollouts for
  step 3 only need ~6144 — set `MAX_MODEL_LEN=6144` there to speed up
  compilation substantially).
- **Instance won't launch: `VcpuLimitExceeded`** — quota (§2) not granted
  yet, or granted in a different region.
- **`neuron-ls` shows nothing** — wrong AMI (must be a *Neuron* DLAMI) or
  wrong instance family.

## 11. Tear down

1. Back up checkpoints/data to S3 (§9).
2. EC2 → select instance → **Instance state → Terminate**.
3. EC2 → **Volumes** — confirm the EBS volume was deleted (it is by default
   when the instance terminates; delete manually if you unchecked that).
4. Billing console → check for stray charges after 24h.
