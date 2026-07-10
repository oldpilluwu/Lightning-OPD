# SPDX-License-Identifier: Apache-2.0

"""
Step 2 (Trainium): SFT training with optimum-neuron, replacing LlamaFactory.

Faithful to configs/sft/qwen3-{4b,8b}-base-open-thoughts3-*.yaml:
  * full fine-tune, bf16
  * qwen3 chat template; loss only on assistant response tokens
  * packing: examples concatenated into fixed blocks of --cutoff-len (16384),
    naive packing (no cross-example attention masking) exactly like
    LlamaFactory's `packing: true` without `neat_packing`
  * lr 8e-5, cosine schedule, warmup_ratio 0.1, max_steps 3000
  * global batch: 256 packed blocks (4B) / 128 packed blocks (8B)

Launch with torchrun, one process per NeuronCore (see step2_sft_train.sh).

Written against optimum-neuron >= 0.3 (Qwen3 training support). If the
import of NeuronModelForCausalLM fails on your version, check
https://huggingface.co/docs/optimum-neuron for the current training API.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM


IGNORE_INDEX = -100

DEBUG = os.environ.get("OPD_DEBUG", "0") == "1"
IS_MAIN = int(os.environ.get("RANK", "0")) == 0


def _dbg(*args):
    if DEBUG and IS_MAIN:
        print("[SFT][DEBUG]", *args, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="e.g. Qwen/Qwen3-4B-Base")
    p.add_argument("--dataset-parquet", required=True,
                   help="Merged SFT parquet from step 1 (column: messages).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cutoff-len", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=8e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=4,
                   help="4 = one trn2.3xlarge chip (4 logical cores); 8 for trn1.32xlarge.")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=10)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--preproc-workers", type=int, default=16)
    p.add_argument("--cache-dir", default="data/sft_cache")
    return p.parse_args()


def build_packed_dataset(args, tokenizer):
    """Tokenize conversations and pack them into fixed cutoff_len blocks.

    Token layout matches how the data was generated in step 1: the prompt is
    rendered with add_generation_prompt=True (thinking enabled) and the
    assistant text + <|im_end|> continues it verbatim, so training sequences
    are token-identical to the teacher's generation trace.
    """
    cache_path = Path(args.cache_dir) / (
        f"{Path(args.dataset_parquet).stem}-{Path(args.model_id).name}-packed{args.cutoff_len}"
    )
    if cache_path.exists():
        from datasets import load_from_disk
        print(f"[SFT] Using cached packed dataset: {cache_path}")
        return load_from_disk(str(cache_path))

    raw = load_dataset("parquet", data_files=args.dataset_parquet, split="train")
    im_end = "<|im_end|>\n"

    if DEBUG and IS_MAIN and len(raw) > 0:
        # Reproduce the tokenization of one example single-threaded so we can
        # show the prompt/response boundary and confirm the label mask lines up
        # with the response tokens (the usual place a template mismatch bites).
        ex = raw[0]
        _dbg(f"raw rows: {len(raw)}; columns: {raw.column_names}")
        msgs = list(ex["messages"])
        _dbg(f"example roles: {[m['role'] for m in msgs]}")
        um = [m for m in msgs if m["role"] != "assistant"]
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        p_ids = tokenizer.apply_chat_template(
            um, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        r_ids = tokenizer.encode((asst or "") + im_end, add_special_tokens=False)
        _dbg(f"prompt tokens={len(p_ids)}, response tokens={len(r_ids)}, "
             f"total={len(p_ids) + len(r_ids)} (cutoff={args.cutoff_len})")
        _dbg(f"prompt ends with: {tokenizer.decode(p_ids[-24:])!r}")
        _dbg(f"response starts with: {tokenizer.decode(r_ids[:24])!r}")
        _dbg(f"response ends with: {tokenizer.decode(r_ids[-12:])!r}")
        # Labels: IGNORE over the prompt, real ids over the response. Verify the
        # first supervised position is exactly the first response token.
        labels = [IGNORE_INDEX] * len(p_ids) + r_ids
        first_sup = next((i for i, x in enumerate(labels) if x != IGNORE_INDEX), None)
        _dbg(f"first supervised index={first_sup} (== prompt len {len(p_ids)} means "
             f"loss starts exactly at the response) ; supervised tokens="
             f"{sum(1 for x in labels if x != IGNORE_INDEX)}")

    def tokenize_fn(batch):
        input_ids_out, labels_out = [], []
        for messages in batch["messages"]:
            messages = list(messages)
            user_messages = [m for m in messages if m["role"] != "assistant"]
            assistant = next((m["content"] for m in messages if m["role"] == "assistant"), None)
            if assistant is None:
                continue
            prompt_ids = tokenizer.apply_chat_template(
                user_messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=True,
            )
            response_ids = tokenizer.encode(assistant + im_end, add_special_tokens=False)
            ids = prompt_ids + response_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
            if len(ids) > args.cutoff_len:
                # Match LlamaFactory's packed SFT: TRUNCATE to cutoff_len (keep
                # the prompt, trim the response tail) rather than dropping the
                # example. LlamaFactory encodes via infer_seqlen, which for this
                # single-turn / short-prompt data truncates the response first;
                # right-truncating the concatenation reproduces that. Dropping
                # would silently shrink the dataset (badly so at a lowered
                # cutoff_len, where LlamaFactory would still keep+truncate).
                ids = ids[:args.cutoff_len]
                labels = labels[:args.cutoff_len]
            input_ids_out.append(ids)
            labels_out.append(labels)
        return {"input_ids": input_ids_out, "labels": labels_out}

    tokenized = raw.map(
        tokenize_fn, batched=True, remove_columns=raw.column_names,
        num_proc=args.preproc_workers, desc="Tokenizing",
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    block = args.cutoff_len

    def pack_fn(batch):
        packed_ids, packed_labels, packed_mask = [], [], []
        cur_ids, cur_labels = [], []
        for ids, labels in zip(batch["input_ids"], batch["labels"]):
            if len(cur_ids) + len(ids) > block:
                pad = block - len(cur_ids)
                packed_ids.append(cur_ids + [pad_id] * pad)
                packed_labels.append(cur_labels + [IGNORE_INDEX] * pad)
                packed_mask.append([1] * len(cur_ids) + [0] * pad)
                cur_ids, cur_labels = [], []
            cur_ids += ids
            cur_labels += labels
        if cur_ids:
            pad = block - len(cur_ids)
            packed_ids.append(cur_ids + [pad_id] * pad)
            packed_labels.append(cur_labels + [IGNORE_INDEX] * pad)
            packed_mask.append([1] * len(cur_ids) + [0] * pad)
        return {"input_ids": packed_ids, "labels": packed_labels,
                "attention_mask": packed_mask}

    packed = tokenized.map(
        pack_fn, batched=True, batch_size=1024,
        remove_columns=tokenized.column_names,
        num_proc=args.preproc_workers, desc="Packing",
    )
    packed = packed.shuffle(seed=42)

    if DEBUG and IS_MAIN and len(packed) > 0:
        # Padding / supervision stats across the (small, in smoke) packed set:
        # high pad fraction => cutoff too long for the data; low supervised
        # fraction => most tokens are prompt (expected) or masking is wrong.
        real = sup = 0
        n = min(len(packed), 256)
        for i in range(n):
            b = packed[i]
            real += sum(b["attention_mask"])
            sup += sum(1 for x in b["labels"] if x != IGNORE_INDEX)
        tot = n * block
        _dbg(f"packed blocks={len(packed)} (stats over first {n}); "
             f"real tokens={real}/{tot} ({real / tot:.1%}), "
             f"pad={1 - real / tot:.1%}, supervised={sup}/{tot} ({sup / tot:.1%})")
        one_step = args.global_batch_size
        verdict = ("OK" if len(packed) >= one_step else
                   "TOO FEW BLOCKS — dataloader_drop_last will drop everything and "
                   "training will do 0 steps; lower GBS or CUTOFF_LEN")
        _dbg(f"blocks/optimizer-step (global batch) = {one_step}; {verdict}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    packed.save_to_disk(str(cache_path))
    print(f"[SFT] Packed dataset: {len(packed)} blocks of {block} tokens -> {cache_path}")
    return packed


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_packed_dataset(args, tokenizer)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size % args.tensor_parallel_size == 0, (
        f"WORLD_SIZE ({world_size}) must be divisible by --tensor-parallel-size "
        f"({args.tensor_parallel_size}); otherwise the data-parallel layout is wrong."
    )
    dp_size = world_size // args.tensor_parallel_size
    denom = dp_size * args.per_device_batch_size
    assert args.global_batch_size % denom == 0, (
        f"--global-batch-size ({args.global_batch_size}) must be divisible by "
        f"dp_size * per_device_batch_size ({dp_size} * {args.per_device_batch_size} = {denom}); "
        f"otherwise gradient accumulation floors and the effective global batch "
        f"silently differs from {args.global_batch_size}."
    )
    grad_accum = args.global_batch_size // denom

    training_args = NeuronTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        tensor_parallel_size=args.tensor_parallel_size,
        zero_1=True,
        dataloader_drop_last=True,
        report_to=["wandb"]
        if os.environ.get("WANDB_API_KEY") and importlib.util.find_spec("wandb")
        else [],
        run_name=Path(args.output_dir).name,
    )

    model = NeuronModelForCausalLM.from_pretrained(
        args.model_id,
        training_args.trn_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    trainer = NeuronTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    ckpts = sorted(Path(args.output_dir).glob("checkpoint-*")) if Path(args.output_dir).exists() else []
    trainer.train(resume_from_checkpoint=bool(ckpts))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
