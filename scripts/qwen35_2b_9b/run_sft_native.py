#!/usr/bin/env python3

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


IGNORE_INDEX = -100


def as_messages(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [dict(item) for item in value]


def render_prompt(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_full(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": False}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class SFTDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, cutoff_len):
        self.rows = pd.read_parquet(parquet_path).to_dict("records")
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        messages = as_messages(self.rows[index]["messages"])
        assistant_idx = next((i for i, msg in enumerate(messages) if msg.get("role") == "assistant"), None)
        if assistant_idx is None:
            raise ValueError(f"Row {index} has no assistant message")

        prompt_messages = messages[:assistant_idx]
        prompt_text = render_prompt(self.tokenizer, prompt_messages)
        full_text = render_full(self.tokenizer, messages)
        if self.tokenizer.eos_token and not full_text.endswith(self.tokenizer.eos_token):
            full_text += self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=self.cutoff_len,
            truncation=True,
        )["input_ids"]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
        if all(label == IGNORE_INDEX for label in labels):
            labels[-1] = full_ids[-1]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch, pad_token_id):
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        size = item["input_ids"].numel()
        input_ids[row, :size] = item["input_ids"]
        labels[row, :size] = item["labels"]
        attention_mask[row, :size] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def sorted_checkpoints(output_dir):
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            pass
    return [path for _, path in sorted(checkpoints)]


def prune_checkpoints(output_dir, save_total_limit):
    if save_total_limit <= 0:
        return
    checkpoints = sorted_checkpoints(output_dir)
    for path in checkpoints[:-save_total_limit]:
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(model, tokenizer, output_dir, step, save_total_limit):
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt, safe_serialization=True)
    tokenizer.save_pretrained(ckpt)
    prune_checkpoints(output_dir, save_total_limit)
    print(f"saved {ckpt}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.train()
    model.to("cuda")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = SFTDataset(args.train_parquet, tokenizer, args.cutoff_len)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    warmup_steps = math.ceil(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)

    state_path = output_dir / "native_sft_state.json"
    state_path.write_text(json.dumps({"max_steps": args.max_steps, "rows": len(dataset)}, indent=2) + "\n")

    step = 0
    accum = 0
    running_loss = 0.0
    progress = tqdm(total=args.max_steps, desc="native sft")
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            accum += 1

            if accum % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{running_loss / args.gradient_accumulation_steps:.4f}")
                running_loss = 0.0

                if step % args.save_steps == 0:
                    save_checkpoint(model, tokenizer, output_dir, step, args.save_total_limit)

                if step >= args.max_steps:
                    break

    progress.close()
    save_checkpoint(model, tokenizer, output_dir, step, args.save_total_limit)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
