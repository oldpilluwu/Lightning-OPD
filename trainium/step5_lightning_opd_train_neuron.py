# SPDX-License-Identifier: Apache-2.0

"""
Step 5 (Trainium): Lightning OPD training with optimum-neuron, replacing the
slime (Megatron + Ray + SGLang) stack.

Faithful port of the training semantics in
configs/lightning_opd/qwen3-{4b,8b}-lightning-opd.py +
slime/backends/megatron_utils/loss.py (`advantage_estimator ==
"on_policy_distillation"`):

  * data: precomputed parquet rows {prompt, metadata.{response_tokens,
    loss_mask, teacher_log_probs}} — student rollouts replayed from disk,
    no rollout engine needed (that is the whole point of Lightning OPD)
  * per response token t:  advantage(t) = logP_teacher(t) - logP_student(t)
    with logP_student taken from the current policy (detached), i.e. the
    reverse-KL policy gradient:  loss = -mean_t[ adv(t) * logP_student(t) ]
    In slime this runs through the PPO surrogate, but with
    rollout_batch_size == global_batch_size == 256 and 1 update per batch the
    importance ratio is exactly 1, so the gradient is identical.
  * kl_loss_coef = 0 and entropy_coef = 0 in the original configs -> omitted
  * optimizer: Adam lr 2e-6 constant, betas (0.9, 0.98), weight decay 0.1
  * global batch 256 sequences, max response length 4096
  * ~150 steps for convergence (README); --max-steps to change

XLA note: every sample is padded to a single static --max-seq-len so the
graph compiles once.

TP note: with tensor_parallel_size > 1 the LM head is vocab-sharded; token
logprobs are then computed with a distributed logsumexp over the TP group
(mathematically identical to the single-device path).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

from logprob_utils import token_logprobs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-checkpoint", required=True,
                   help="Consolidated HF checkpoint from step 2.")
    p.add_argument("--data-parquet", required=True,
                   help="Precomputed parquet from step 4 (*-precomputed.parquet).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=5632,
                   help="Static padded length; must be >= prompt + response (4096).")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--save-total-limit", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    # Optional OPD probe (same probe set as step 2) to watch the OPD stage
    # close the teacher gap that SFT plateaued at.
    p.add_argument("--opd-probe-parquet", default=None)
    p.add_argument("--probe-every", type=int, default=5)
    p.add_argument("--probe-size", type=int, default=64)
    p.add_argument("--probe-max-seq-len", type=int, default=4608)
    p.add_argument("--probe-micro-batch", type=int, default=1)
    return p.parse_args()


class LightningOPDDataset(Dataset):
    """Replays precomputed student rollouts with teacher logprobs.

    Produces fixed-shape tensors:
      input_ids      [L]  prompt + response + pad
      attention_mask [L]
      response_mask  [L]  1 where the token is a *response* token (loss_mask
                          from the parquet, respecting its 0/1 values)
      teacher_lp     [L]  teacher logprob of the token at that position
    """

    def __init__(self, parquet_path, tokenizer, max_seq_len, pad_id, seed):
        df = pd.read_parquet(parquet_path)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))  # --rollout-shuffle equivalent

        self.samples = []
        dropped = 0
        for i in order:
            row = df.iloc[int(i)]
            meta = row["metadata"]
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            response_ids = [int(x) for x in meta["response_tokens"]]
            loss_mask = [int(x) for x in meta["loss_mask"]]
            teacher_lp = [float(x) for x in meta["teacher_log_probs"]]
            total = len(prompt_ids) + len(response_ids)
            if total > max_seq_len:
                dropped += 1
                continue
            self.samples.append((prompt_ids, response_ids, loss_mask, teacher_lp))
        if dropped:
            print(f"[OPD data] Dropped {dropped}/{len(df)} rows longer than {max_seq_len}")
        print(f"[OPD data] {len(self.samples)} training samples")

        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_ids, response_ids, loss_mask, teacher_lp = self.samples[idx]
        L = self.max_seq_len
        n_p, n_r = len(prompt_ids), len(response_ids)

        input_ids = prompt_ids + response_ids + [self.pad_id] * (L - n_p - n_r)
        attention_mask = [1] * (n_p + n_r) + [0] * (L - n_p - n_r)
        response_mask = [0] * n_p + loss_mask + [0] * (L - n_p - n_r)
        t_lp = [0.0] * n_p + teacher_lp + [0.0] * (L - n_p - n_r)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.float32),
            "teacher_lp": torch.tensor(t_lp, dtype=torch.float32),
        }


class LightningOPDTrainer(NeuronTrainer):
    def __init__(self, *args, vocab_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        response_mask = inputs["response_mask"]
        teacher_lp = inputs["teacher_lp"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Position i predicts token i+1: align everything to "target token" space
        student_lp = token_logprobs(logits[:, :-1, :], input_ids[:, 1:], self._vocab_size)
        mask = response_mask[:, 1:]
        t_lp = teacher_lp[:, 1:]

        # advantage = logP_teacher - logP_student  (detached), from
        # slime loss.py `advantage_estimator == "on_policy_distillation"`
        advantage = (t_lp - student_lp).detach()
        pg_loss = -(advantage * student_lp)

        denom = mask.sum().clamp(min=1.0)
        loss = (pg_loss * mask).sum() / denom
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    dataset = LightningOPDDataset(
        args.data_parquet, tokenizer, args.max_seq_len, pad_id, args.seed
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_size = world_size // args.tensor_parallel_size
    grad_accum = max(1, args.global_batch_size // (dp_size * args.per_device_batch_size))

    training_args = NeuronTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        tensor_parallel_size=args.tensor_parallel_size,
        zero_1=True,
        dataloader_drop_last=True,
        seed=args.seed,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=Path(args.output_dir).name,
    )

    model = NeuronModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        training_args.trn_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    vocab_size = model.config.vocab_size

    callbacks = []
    if args.opd_probe_parquet:
        from opd_probe import attach_probe
        attach_probe(
            callbacks,
            probe_parquet=args.opd_probe_parquet,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            output_dir=args.output_dir,
            every=args.probe_every,
            probe_size=args.probe_size,
            max_seq_len=args.probe_max_seq_len,
            micro_batch=args.probe_micro_batch,
            tag="opd_probe",
        )

    trainer = LightningOPDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        vocab_size=vocab_size,
        callbacks=callbacks,
    )

    ckpts = sorted(Path(args.output_dir).glob("checkpoint-*")) if Path(args.output_dir).exists() else []
    trainer.train(resume_from_checkpoint=bool(ckpts))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
