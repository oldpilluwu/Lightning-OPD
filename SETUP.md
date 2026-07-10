# Lightning OPD on AWS Trainium — Setup Guide

This guide takes you from a fresh AWS account to a trained Lightning-OPD model
on AWS Trainium, using the scripts in [`trainium/`](trainium/). It targets a
single **`trn2.3xlarge`** (one Trainium2 chip) running the **Qwen3-4B** scale.

> **Scale note.** The **4B** scale (student Qwen3-4B-Base, teacher Qwen3-8B)
> fits on one trn2.3xlarge chip. The **8B** scale does **not**: a Qwen3-8B
> full fine-tune needs ~128 GB and one chip has only 96 GiB of HBM, and a
> single chip has no data-parallel dimension to shard optimizer state across
> (raising tensor-parallel size splits tensors within the *same* HBM pool, so
> it does not add capacity). For 8B, use a larger instance (e.g.
> trn1.32xlarge / trn2.48xlarge with `NUM_CORES`/`TRAIN_TP` raised) — the same
> scripts scale up — or a parameter-efficient (LoRA) variant, which is not
> wired up here yet.

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
> | 4. Teacher logprobs | sglang server + HTTP | **forward-pass scoring** through `NeuronModelForCausalLM` — `trainium/step4_teacher_forward_neuron.py` (same quantity, same parquet schema). *vLLM `prompt_logprobs` — `step4_teacher_logprobs_neuron.py` — is broken on vllm-neuron 0.16 and kept only as `BACKEND=vllm`.* |
> | 5. Lightning OPD | slime (Megatron+Ray) | custom `NeuronTrainer` loss — `trainium/step5_lightning_opd_train_neuron.py` (advantage = teacher logprob − student logprob, lr 2e-6, batch 256, identical) |
> | 6. Ckpt convert | Megatron→HF script | `optimum-cli neuron consolidate` (built into the pipeline) |
>
> These scripts were written against **Neuron SDK ≥ 2.24** (Qwen3 support in
> NxD Inference) and **optimum-neuron ≥ 0.3.0** (Qwen3 training support) and
> have **not been validated on real Trainium hardware** — run the smoke test
> (§7) before committing to the full run. On **Trainium2 (trn2.x)** use a
> recent SDK; the defaults assume the single-chip `trn2.3xlarge` (4 logical
> NeuronCores, 96 GiB) and the 4B scale — see §1 for scaling up and for why
> 8B needs a larger instance. The known risk points are in §10.

---

## 1. Choose instance type and region

Everything (4B scale) runs on **one `trn2.3xlarge`**: a single Trainium2 chip
with **8 physical NeuronCores exposed as 4 logical NeuronCores** (the default
LNC=2 config groups two physical cores into one logical core), **96 GiB HBM**,
and ~24 vCPUs. Confirm the core count on the instance with `neuron-ls`.

- Because the chip presents **4 logical cores**, the scripts default to
  `NUM_CORES=4` and tensor-parallel size `4` (one worker uses the whole chip;
  there is no data-parallel dimension). Do not set these above 4 on this
  instance.
- Regions with Trn2: **us-east-1 (N. Virginia)**, **us-east-2 (Ohio)**,
  **us-west-2 (Oregon)**. Pick one and stay in it. Trn2 quota is often only
  available via **EC2 Capacity Blocks for ML** — check availability before
  planning a long run.
- Scaling up: the same scripts run on **trn1.32xlarge** (32 logical cores,
  512 GB) or **trn2.48xlarge** (64 logical cores) — set `NUM_CORES` and
  `TRAIN_TP`/`GEN_TP` accordingly. This is required for the **8B** scale,
  which does not fit on one chip (see the scale note at the top).
- Do **not** try `trn1.2xlarge` — a single Trainium1 chip (2 cores × 16 GB)
  cannot hold even the 4B model with optimizer state.

Rough budget for the **4B** scale on one trn2.3xlarge (on-demand; the dominant
cost is step 1 generating 300k × up-to-16k-token responses). A single chip is
a small slice of a full node, so the full-scale run is slow — the smoke test
(§7) or a reduced `SFT_SAMPLES`/`OPD_STEPS` run is the practical path:

| Stage | Wall-clock (very rough, 1 chip) |
|---|---|
| Step 1 (300k SFT gen) | days — reduce `SFT_SAMPLES` for a real budget |
| Step 2 (SFT) | many hours–days at 3000 steps |
| Steps 3–4 (rollouts + teacher scoring) | hours |
| Step 5 (OPD, ~150 steps) | hours |

The paper's numbers (20/30 GPU-hours) are for the OPD stage only, on 8×H100.
If you only want to validate the pipeline, run the smoke test first (§7,
64 SFT / 64 rollouts / 8 steps each, real config on a small dataset).

## 2. Request quota (AWS Console)

New accounts have **zero** Trn quota. Do this first — approval can take a
few hours to a couple of days.

1. Sign in to the AWS Console, set your region (top-right) to e.g. **us-east-2**.
2. Search bar → **Service Quotas** → **AWS services** → **Amazon Elastic
   Compute Cloud (Amazon EC2)**.
3. Search the quota list for **"Running On-Demand Trn instances"**.
4. **Request increase at account level** → enter enough vCPUs for the
   instance you want (a **trn2.3xlarge** uses ~24; a trn1.32xlarge uses 128).
   Submit.
5. Watch status under *Quota request history*. If it stalls, open a Support
   case (Support Center → Create case → Service limit increase) and mention
   you're running an ML training job. Note: Trn2 capacity is frequently only
   offered through **EC2 Capacity Blocks for ML** rather than plain
   on-demand — if on-demand launch fails, reserve a Capacity Block instead.

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
2. **Name**: `lightning-opd-trn2`.
3. **Application and OS Images (AMI)** → *Browse more AMIs* → search
   **"Deep Learning AMI Neuron"** → pick
   **Deep Learning AMI Neuron (Ubuntu 22.04)** (the multi-framework Neuron
   DLAMI; it ships the Neuron driver, runtime, and the PyTorch/NxD-Inference
   virtualenvs the scripts expect). Make sure its Neuron SDK is recent enough
   for Trainium2 + Qwen3 (SDK ≥ 2.24; newer is better for Trn2 support).
4. **Instance type**: `trn2.3xlarge`.
5. **Key pair**: `trainium-key`.
6. **Network settings** → Edit:
   - Keep the default VPC/subnet, **Auto-assign public IP: Enable**.
   - Security group: *Create security group*, allow **SSH (22)** with source
     **My IP** only.
7. **Configure storage**: set the root volume to **1000 GiB, gp3** (checkpoints
   + Neuron compile cache + a reduced SFT dataset; go to **2000 GiB** if you
   run the full 300k-sample generation). Optionally raise gp3 throughput to
   500 MB/s.
8. **Launch instance**. Wait until *Instance state = Running* and note the
   **Public IPv4 address** from the instance details page.

> 💡 Cost control: **Stop** the instance (EC2 → Instance state → Stop)
> whenever you're not training — the EBS volume (and all your data) persists
> across stop/start; only the public IP changes. The resumable pipeline picks
> up where it left off. (If you reserved a Capacity Block, note it bills for
> the whole reserved window regardless of stop/start.)

## 5. Connect and set up

```bash
ssh -i ~/Downloads/trainium-key.pem ubuntu@<PUBLIC_IP>
```

Verify the accelerator — on a trn2.3xlarge you should see **1 device**, and
under the default LNC=2 config **4 logical NeuronCores** (this is the number
the scripts use for `NUM_CORES`). If `neuron-ls` reports a different logical
core count, set `NUM_CORES`/`TRAIN_TP`/`GEN_TP` to match it.

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

### 5a. Giving teammates access (shared `ubuntu` account)

For a small trusted team it's simplest for everyone to share the default
`ubuntu` account and authenticate with their own SSH keys — no per-user
accounts, no IAM needed. Two steps:

**1. Open SSH to the team in the security group.** EC2 → select the instance →
**Security** tab → click the security group → **Edit inbound rules** → the
SSH/22 rule → set the **Source** to your VPN/office CIDR (ask your admin, e.g.
`10.0.0.0/16`), or add one rule per teammate IP as `x.x.x.x/32`. Avoid
`0.0.0.0/0` unless you must — this is an expensive box and password auth should
stay disabled (it is by default on the DLAMI; keep it that way).

**2. Add each teammate's public key.** Each teammate generates a key on *their*
machine and sends you the **public** half only (never the private key):

```bash
ssh-keygen -t ed25519 -C "alice@org"
cat ~/.ssh/id_ed25519.pub        # send this one line
```

You (logged in with the launch key as `ubuntu`) append them all to the shared
`authorized_keys`:

```bash
cat >> ~/.ssh/authorized_keys <<'EOF'
ssh-ed25519 AAAA...alice... alice@org
ssh-ed25519 AAAA...bob...   bob@org
EOF
```

Everyone then connects as the same user with their own key:

```bash
ssh ubuntu@<PUBLIC_IP>
```

What sharing one account means in practice:

- The `-C "alice@org"` **key comment is the only way to tell whose key is
  whose** — keep it meaningful. To revoke someone, delete their line.
- **No per-person audit** — everything runs as `ubuntu`, and everyone has that
  user's passwordless sudo, so anyone can modify or kill anyone else's runs.
  Fine for a trusted team; not for anything sensitive.
- **Always run training inside a *shared* `tmux`** so a dropped SSH session
  doesn't kill a multi-day job and teammates can attach to the same view:
  ```bash
  tmux new -s opd          # first person starts it
  tmux attach -t opd       # everyone else attaches to the same session
  ```

## 6. Smoke-test the two risk points (15–30 min, do not skip)

> **Run these smoke tests from a FILE with an `if __name__ == "__main__":`
> guard — not a `python - <<'PY'` heredoc and not bare module-level code.** vLLM
> V1 launches its `EngineCore` in a separate process via `multiprocessing`
> spawn, and the child re-imports the main module. A heredoc has no file to
> re-import (`FileNotFoundError: …/<stdin>`), and a bare top-level `LLM(...)`
> re-runs during that import and tries to spawn again (`An attempt has been made
> to start a new process before the current process has finished its
> bootstrapping phase`). Putting the engine call inside `main()` under the
> `__main__` guard (below) fixes both. The pipeline scripts already do this, so
> they're unaffected. (Quick interactive alternative:
> `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs the engine in-process.)

**(a) Qwen3 generation on Neuron** — verifies the vLLM Neuron plugin +
Qwen3 support:

```bash
# Use the pre-installed vLLM venv (name is auto-detected by the scripts; find it
# with: ls /opt | grep vllm). On the current DLAMI it is:
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
cat > ~/smoke_gen.py <<'PY'
from vllm import LLM, SamplingParams

def main():
    # enable_prefix_caching=False: V1 defaults it ON, which then demands an
    # explicit block_size; batch gen doesn't benefit (prompts ~unique), so off.
    llm = LLM(model="Qwen/Qwen3-0.6B", tensor_parallel_size=2, max_model_len=2048,
              max_num_seqs=4, enable_prefix_caching=False)
    out = llm.chat([[{"role": "user", "content": "What is 2+2?"}]], SamplingParams(max_tokens=64))
    print(out[0].outputs[0].text)

if __name__ == "__main__":   # REQUIRED — vLLM V1 spawns a subprocess
    main()
PY
python ~/smoke_gen.py
```

**(b) Teacher scoring (step 4).** Step 4 defaults to `BACKEND=forward`
(`step4_teacher_forward_neuron.py`) — a plain forward pass through
`NeuronModelForCausalLM`, no vLLM. The smoke run exercises it, so there is no
separate one-liner to run here; just confirm the end-to-end smoke run (§7)
completes step 4.

> **Why not vLLM `prompt_logprobs`?** We tried it on this DLAMI
> (vllm-neuron 0.16) and it is **broken**: the default on-device sampler returns
> no logprobs (`[None]`), and the CPU-sampling path
> (`NEURON_ON_DEVICE_SAMPLING_DISABLED=1`) crashes — first a `Sampler` import bug
> (`from vllm.v1.sample import sampler as Sampler` → `Sampler()` →
> `'module' object is not callable`), then, after patching that, the model runner
> still routes to `_sample_on_device` → `hidden_states[reorder_indices]` →
> `IndexError`. The `BACKEND=vllm` path is retained only for future SDKs where
> `prompt_logprobs` works; if you want to re-check it, use the [vLLM V1 NxDI
> guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html).

You *can* smoke the forward scorer directly on a tiny slice once you have an SFT
checkpoint and a Phase-1 intermediate parquet:

```bash
# in the TRAIN venv (optimum-neuron), from the repo root
NUM_CORES=4 TP_SIZE=4 torchrun --nproc_per_node=4 \
  trainium/step4_teacher_forward_neuron.py \
  --teacher-model Qwen/Qwen3-8B --tokenizer-path <sft_ckpt> \
  --intermediate-parquet <stem>-lightning-opd.parquet \
  --output-parquet /tmp/probe-precomputed.parquet \
  --num-rows 8 --max-seq-len 8192
```

It prints the kept-logprob count and per-chunk mean/min/max (all logprobs
should be ≤ 0).

## 7. Run the pipeline

Always inside `tmux` so an SSH drop doesn't kill a multi-day run:

```bash
tmux new -s opd
cd ~/Lightning-OPD
```

**Recommended first: end-to-end smoke run** — a *faithful mini-run*. It uses
the **same config as the real run** (generation length 16384, SFT packing
`CUTOFF_LEN=16384`, OPD sequence length 5632, learning rates, schedules,
warmup, betas all unchanged), so it compiles the same Neuron graphs, hits the
same memory footprint, and exercises the same code paths. Only the **dataset**
is smaller — 64 SFT prompts / 64 rollouts — and the step count is short
(8 SFT + 8 OPD steps). The one unavoidable deviation is the global batch size:
a 256-sequence batch can't be built from 64 samples, so `SFT_GBS`/`OPD_GBS` are
scaled to 8 (this changes only the gradient-accumulation count, not any
compiled shape or hyperparameter):

```bash
SCALE=4b SMOKE=1 bash trainium/run_pipeline.sh
```

Because the shapes match the real run, if step 2 or 5 OOMs in the smoke run it
will OOM in the full run too — lower `CUTOFF_LEN` / `MAX_SEQ_LEN` for both.

The smoke run turns on `OPD_DEBUG=1`, which prints diagnostics tagged
`[DEBUG]` at each step so you can spot problems without hardware traces:
prompt/chat structure and a sample generation (steps 1/3); the
prompt→response boundary, label masking, and packed-block padding/supervision
stats (step 2); teacher-logprob alignment and distribution stats (step 4); and
dataset alignment plus a one-time student-vs-teacher logprob/advantage dump
inside the loss (step 5). Set `OPD_DEBUG=1` on a full run too if you want the
same output. Smoke knobs: `SFT_GBS` / `OPD_GBS` (global batch, default 8),
`SFT_STEPS` / `OPD_STEPS` (default 8). Everything else tracks the real-run
defaults below.

**Full 4B reproduction** (student Qwen3-4B-Base, teacher Qwen3-8B) — the
supported single-chip path:

```bash
SCALE=4b bash trainium/run_pipeline.sh
```

**8B scale** (student Qwen3-8B-Base, teacher Qwen3-32B): does **not** fit on
one trn2.3xlarge chip (see the scale note in §1) — `run_pipeline.sh` refuses
it with `NUM_CORES<=4`. Run it on a larger instance instead, e.g. a
trn1.32xlarge:

```bash
NUM_CORES=32 TRAIN_TP=8 GEN_TP=8 SCALE=8b bash trainium/run_pipeline.sh
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
| `NUM_CORES` | 4 | Logical NeuronCores (4 = one trn2.3xlarge chip; 32 for trn1.32xlarge) |
| `TRAIN_TP` | 4 | Tensor-parallel size for SFT/OPD (must divide `NUM_CORES`) |
| `GEN_TP` | `NUM_CORES` | NeuronCores per vLLM worker in generation/scoring |
| `SFT_SAMPLES` | 300000 | SFT prompts sampled from OpenThoughts3-1.2M (reduce on 1 chip) |
| `SFT_STEPS` | 3000 | SFT optimizer steps (paper value) |
| `OPD_STEPS` | 150 | Lightning OPD steps (README: ~150 converges; paper config caps at 3000) |
| `SFT_GBS` | 256 (4b) / 128 (8b) | SFT global batch (matches LlamaFactory configs) |
| `CUTOFF_LEN` | 16384 (paper; unchanged in smoke) | SFT packed sequence length — main OOM lever on one chip; drop to 8192/4096 if step 2 OOMs |
| `OPD_DEBUG` | 0 (1 in smoke) | Print `[DEBUG]` diagnostics at every step (prompt/label alignment, dataset structure, logprob stats) |

## 8. Choosing the SFT checkpoint

Run SFT for the paper's step count (`SFT_STEPS`, default 3000) and use the
resulting checkpoint for steps 3–5. `run_pipeline.sh` consolidates the final
SFT checkpoint automatically; if you want an earlier one, any
`checkpoints/<sft-dir>/checkpoint-<step>` (written every `save_steps`) can be
consolidated the same way. Watch the ordinary SFT train loss for a smooth,
plateauing curve; if it is still falling steadily at step 3000, extend
`SFT_STEPS`.

## 9. S3 for datasets and checkpoints

The pipeline reads and writes the instance's **local** `data/` and
`checkpoints/` directories — it does not talk to S3 itself. S3 is used
alongside it, for two things: **seeding/sharing the dataset** and **backing up
checkpoints** so nothing is lost when the instance is stopped or terminated
(and so teammates can pull the results).

### 9a. Create a bucket (once)

From any machine with AWS access, or from the instance once credentials are set
(9b):

```bash
aws s3 mb s3://<your-org>-lightning-opd --region us-east-2   # match your instance region
```

Bucket names are globally unique — prefix with your org/username. Keep the
bucket in the **same region** as the instance so transfers are fast and free of
cross-region egress cost.

### 9b. Give the instance credentials

The AWS CLI is preinstalled on the DLAMI but has no credentials by default.
Two ways:

- **IAM role (preferred, no keys on disk)** — EC2 → select instance → Actions →
  Security → **Modify IAM role** → attach a role granting S3 access to this
  bucket. Nothing else to configure; `aws s3` just works.
- **Access key (if you can't create/attach a role)** — from your AWS console,
  **Security credentials → Create access key**, then on the instance:
  ```bash
  aws configure    # paste Access Key ID, Secret, region (us-east-2), output=json
  ```
  This writes `~/.aws/credentials`. On a **shared instance** (§5a) this key is
  readable by every teammate using the `ubuntu` account, so scope it to just
  this bucket and rotate/delete it when you tear down. Verify with:
  ```bash
  aws s3 ls s3://<your-org>-lightning-opd
  ```

### 9c. Dataset — push once, pull on each new instance

If you've already generated/curated the SFT data (steps 0–1) and don't want to
regenerate it (the dominant cost — days on one chip), upload it and re-pull it
on any future instance:

```bash
# after generating, from the instance:
aws s3 sync data/ s3://<your-org>-lightning-opd/data/

# on a fresh instance, before running the pipeline:
cd ~/Lightning-OPD
aws s3 sync s3://<your-org>-lightning-opd/data/ data/
```

Because the pipeline is resumable via the markers under
`data/.pipeline_state/`, restoring `data/` (including those markers) lets a new
instance **skip stages that were already done**. Sync `data/` up periodically
during long runs so a terminated instance doesn't lose completed generation.

### 9d. Checkpoints — back up before terminating

The pipeline ends with a consolidated HuggingFace-format model at:

```
checkpoints/qwen3-4b-lightning-opd-hf/    # or qwen3-8b-...
```

Back it up before terminating anything (and periodically, to survive a crash):

```bash
aws s3 sync checkpoints/ s3://<your-org>-lightning-opd/checkpoints/
```

Teammates or a future instance pull it back with the reverse `sync`. The
checkpoint loads with vLLM/transformers on any hardware — evaluation (AIME /
HMMT / LiveCodeBench) is not part of this repo's pipeline.

> 💡 `aws s3 sync` is incremental (only changed files) and safe to re-run, so
> it's well suited to a periodic `data/` + `checkpoints/` backup during long
> runs — e.g. drop it in a `while true; do aws s3 sync … ; sleep 3600; done`
> in a spare tmux pane.

## 10. Troubleshooting

- **First training step takes 30–60+ min** — that's the Neuron compiler
  tracing and compiling the graphs (once per shape). Subsequent steps are
  fast; the cache persists in `/var/tmp/neuron-compile-cache` across runs.
  You can pre-populate caches for step 2/5 with `neuron_parallel_compile`.
- **Step 4 teacher scoring**: the default `BACKEND=forward`
  (`step4_teacher_forward_neuron.py`) computes teacher logprobs with a forward
  pass through `NeuronModelForCausalLM` + `token_logprobs()` — no vLLM. If it
  errors, the likely spots are (a) `--max-seq-len` too small (it aborts up front
  and prints the value to use), (b) device OOM on a large teacher — lower
  `--batch-size` (already 1) or `MAX_MODEL_LEN`, or raise `TP_SIZE`/`NUM_CORES`
  to shard the teacher across more cores, (c) the model-forward-without-a-Trainer
  or static-padding assumptions needing an SDK-specific tweak (validate on a
  `--num-rows 8` slice first, per §6(b)). The old vLLM path (`BACKEND=vllm`,
  `step4_teacher_logprobs_neuron.py`) is **broken on vllm-neuron 0.16** — see
  §6(b) for the exact bugs; don't use it unless a newer plugin fixes
  `prompt_logprobs`.
- **optimum-neuron trainer imports fail (steps 2/5)**: this DLAMI ships
  optimum-neuron **0.4.3**, whose trainer classes (`NeuronTrainer`,
  `NeuronSFTTrainer`, `NeuronTrainingArguments`) load a PEFT/LoRA shim that
  needs the exact peft/trl it pins (`peft==0.17.0`, `trl==0.24.0`). Without them
  the import fails as `type object 'LoraLinear' has no attribute 'merge'`, and
  optimum-neuron's lazy loader masks it as a misleading "cannot import name …
  Did you mean NeuronSFTTrainer" — the model classes still import, only the
  trainers break. `setup_env.sh` installs these pins; if you hit it, run
  `pip install "peft==0.17.0" "trl==0.24.0"` then re-assert numpy
  (`pip install --force-reinstall "numpy>=2.0.0,<2.5"`). The datasets, loss, and
  hyperparameters don't change. See the finetune-qwen3 guide for API details:
  https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3
- **Device OOM in step 2/5 (training)**: on a single trn2.3xlarge chip,
  raising `TRAIN_TP` does **not** help — with `NUM_CORES=4` there is no
  data-parallel dimension, and TP only splits tensors within the same 96 GiB
  HBM pool. Instead lower `CUTOFF_LEN` (16384 → 8192 → 4096 each halves
  activation memory; changes packing but not the data) for step 2, and lower
  `MAX_SEQ_LEN` for step 5. If 4B full FT still OOMs, the model+optimizer
  footprint is too large for one chip — move to a multi-chip instance
  (`NUM_CORES=32 TRAIN_TP=8`) where ZeRO-1 can shard optimizer state across
  the data-parallel ranks. (On a bigger box, raising `TRAIN_TP` *does* help,
  as does adding DP.)
- **HBM OOM or hangs in generation (steps 1/3/4)**: reduce `MAX_NUM_SEQS`
  (default 8) or `MAX_MODEL_LEN` (default 18432; 4096-token rollouts for
  step 3 only need ~6144 — set `MAX_MODEL_LEN=6144` there to speed up
  compilation substantially). The Qwen3-32B teacher (8B scale) barely fits a
  single chip even at TP=4 — prefer a larger instance for 8B teacher scoring.
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
