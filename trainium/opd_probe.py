# SPDX-License-Identifier: Apache-2.0

"""
OPD probe: periodic student evaluation on a frozen set of teacher responses
to OPD prompts, run during SFT (step 2) and optionally during OPD (step 5).

Metrics per probe (logged to <output_dir>/opd_probe_log.jsonl, stdout, and
wandb when active):

  student_nll  - student per-token NLL on teacher responses to OPD prompts
  teacher_nll  - teacher per-token NLL on its own tokens (constant baseline;
                 the teacher's sampling entropy)
  fwd_kl       - student_nll - teacher_nll = Monte-Carlo estimate of
                 KL(teacher || student) per token on the OPD domain.
                 THE decision metric: SFT is "enough" when it plateaus.
  drift_mean   - mean signed change of student token logprobs since the
                 previous probe (positive = moving toward these tokens)
  drift_abs    - mean |change| of student token logprobs since the previous
                 probe (policy drift magnitude between probes)
  top1_match   - fraction of teacher tokens that are the student's argmax

All ranks execute the probe forward passes (required for tensor-parallel
collectives); metrics are identical on every rank and only the main process
logs. Probe sequences are padded to one static length so XLA compiles the
probe graph once.
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import TrainerCallback

from logprob_utils import token_logprobs


class OPDProbe:
    """Loads a probe parquet (schema of step-4 *-precomputed.parquet) into
    fixed-shape tensors."""

    def __init__(self, parquet_path, tokenizer, max_seq_len=4608, probe_size=None,
                 micro_batch=1, pad_id=None, seed=42):
        df = pd.read_parquet(parquet_path)
        if probe_size is not None and probe_size < len(df):
            df = df.sample(n=probe_size, random_state=seed).reset_index(drop=True)

        pad_id = pad_id if pad_id is not None else (
            tokenizer.pad_token_id or tokenizer.eos_token_id
        )

        ids_rows, am_rows, mask_rows, tlp_rows = [], [], [], []
        dropped = 0
        for _, row in df.iterrows():
            meta = row["metadata"]
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            response_ids = [int(x) for x in meta["response_tokens"]]
            loss_mask = [int(x) for x in meta["loss_mask"]]
            teacher_lp = [float(x) for x in meta["teacher_log_probs"]]
            n_p, n_r = len(prompt_ids), len(response_ids)
            if n_p + n_r > max_seq_len:
                dropped += 1
                continue
            pad = max_seq_len - n_p - n_r
            ids_rows.append(prompt_ids + response_ids + [pad_id] * pad)
            am_rows.append([1] * (n_p + n_r) + [0] * pad)
            mask_rows.append([0] * n_p + loss_mask + [0] * pad)
            tlp_rows.append([0.0] * n_p + teacher_lp + [0.0] * pad)

        # Trim to a multiple of micro_batch so every forward has one shape
        n = (len(ids_rows) // micro_batch) * micro_batch
        if n == 0:
            raise ValueError(
                f"No probe rows fit max_seq_len={max_seq_len} "
                f"(dropped {dropped}); increase --probe-max-seq-len."
            )
        if dropped or n < len(ids_rows):
            print(f"[Probe] Using {n} rows (dropped {dropped} too-long, "
                  f"trimmed {len(ids_rows) - n} to microbatch multiple)")

        self.input_ids = torch.tensor(ids_rows[:n], dtype=torch.long)
        self.attention_mask = torch.tensor(am_rows[:n], dtype=torch.long)
        # Target-token space: position i predicts token i+1
        self.mask = torch.tensor(mask_rows[:n], dtype=torch.float32)[:, 1:]
        self.teacher_lp = torch.tensor(tlp_rows[:n], dtype=torch.float32)[:, 1:]
        self.micro_batch = micro_batch


class OPDProbeCallback(TrainerCallback):
    def __init__(self, probe: OPDProbe, vocab_size: int, every: int,
                 out_jsonl: str, tag: str = "opd_probe"):
        self.probe = probe
        self.vocab_size = vocab_size
        self.every = max(1, every)
        self.out_jsonl = Path(out_jsonl)
        self.tag = tag
        self.prev_lp = None  # student logprobs at the previous probe (CPU)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.every != 0:
            return
        self._run_probe(model, state)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Final probe so the last checkpoint always has a reading
        if model is not None and state.global_step % self.every != 0:
            self._run_probe(model, state)

    @torch.no_grad()
    def _run_probe(self, model, state):
        t0 = time.time()
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        mb = self.probe.micro_batch

        lp_chunks, top1_chunks = [], []
        for i in range(0, self.probe.input_ids.size(0), mb):
            ids = self.probe.input_ids[i:i + mb].to(device)
            am = self.probe.attention_mask[i:i + mb].to(device)
            out = model(input_ids=ids, attention_mask=am)
            logits = out.logits if hasattr(out, "logits") else out[0]
            lp, top1 = token_logprobs(
                logits[:, :-1, :], ids[:, 1:], self.vocab_size, return_top1=True
            )
            lp_chunks.append(lp.float().cpu())
            top1_chunks.append(top1.float().cpu())
        model.train(was_training)

        lp = torch.cat(lp_chunks)          # [N, L-1]
        top1 = torch.cat(top1_chunks)
        mask = self.probe.mask
        n_tok = mask.sum().clamp(min=1.0)

        student_nll = float(-(lp * mask).sum() / n_tok)
        teacher_nll = float(-(self.probe.teacher_lp * mask).sum() / n_tok)
        metrics = {
            "step": state.global_step,
            "student_nll": student_nll,
            "teacher_nll": teacher_nll,
            "fwd_kl": student_nll - teacher_nll,
            "top1_match": float((top1 * mask).sum() / n_tok),
            "probe_tokens": int(n_tok),
            "probe_seconds": round(time.time() - t0, 1),
        }
        if self.prev_lp is not None:
            d = (lp - self.prev_lp) * mask
            metrics["drift_mean"] = float(d.sum() / n_tok)
            metrics["drift_abs"] = float(d.abs().sum() / n_tok)
        self.prev_lp = lp

        if state.is_world_process_zero:
            self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_jsonl, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            drift = (f" drift_abs={metrics['drift_abs']:.4f}"
                     if "drift_abs" in metrics else "")
            print(f"[{self.tag} @ step {state.global_step}] "
                  f"student_nll={student_nll:.4f} teacher_nll={teacher_nll:.4f} "
                  f"fwd_kl={metrics['fwd_kl']:.4f} "
                  f"top1={metrics['top1_match']:.3f}{drift} "
                  f"({metrics['probe_seconds']}s)")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"{self.tag}/{k}": v for k, v in metrics.items()
                               if k != "step"}, step=state.global_step)
            except ImportError:
                pass


def attach_probe(trainer_callbacks: list, *, probe_parquet, tokenizer, vocab_size,
                 output_dir, every=5, probe_size=64, max_seq_len=4608,
                 micro_batch=1, tag="opd_probe"):
    """Convenience builder used by the step 2 / step 5 training scripts."""
    probe = OPDProbe(
        probe_parquet, tokenizer,
        max_seq_len=max_seq_len, probe_size=probe_size, micro_batch=micro_batch,
    )
    cb = OPDProbeCallback(
        probe, vocab_size, every,
        out_jsonl=str(Path(output_dir) / "opd_probe_log.jsonl"), tag=tag,
    )
    trainer_callbacks.append(cb)
    return cb
