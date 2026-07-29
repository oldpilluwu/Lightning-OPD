#!/usr/bin/env python3
"""Shard the Hugging Face Qwen3 checkpoint for the direct TP=8 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

COLUMN_PARALLEL_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
)
ROW_PARALLEL_SUFFIXES = (
    "o_proj.weight",
    "down_proj.weight",
)


def partition_dimension(name: str) -> int | None:
    if name.endswith(COLUMN_PARALLEL_SUFFIXES):
        return 0
    if name.endswith(ROW_PARALLEL_SUFFIXES):
        return 1
    return None


def shard_state_dict(
    full_state: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> dict[str, torch.Tensor]:
    shard: dict[str, torch.Tensor] = {}
    for name, tensor in full_state.items():
        partition_dim = partition_dimension(name)
        if partition_dim is None:
            shard[name] = tensor
            continue
        dimension = tensor.shape[partition_dim]
        if dimension % tp_size:
            raise ValueError(
                f"{name} dimension {partition_dim} has size {dimension}, "
                f"which is not divisible by TP={tp_size}"
            )
        partition_size = dimension // tp_size
        shard[name] = (
            tensor.narrow(
                partition_dim,
                tp_rank * partition_size,
                partition_size,
            )
            .contiguous()
            .clone()
        )
    return shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    pretrained_dir = output_root / "pretrained_weight"
    if (pretrained_dir / "model").exists():
        raise FileExistsError(
            f"{pretrained_dir / 'model'} already exists; refusing to overwrite it"
        )

    config = AutoConfig.from_pretrained(args.model_id)
    if config.model_type != "qwen3":
        raise ValueError(f"Expected a Qwen3 checkpoint, found {config.model_type}")
    if args.tp_size != 8:
        raise ValueError("This Qwen3-4B reproduction is audited for TP=8")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    full_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    del model

    embedding_key = "model.embed_tokens.weight"
    if embedding_key not in full_state:
        raise RuntimeError("Downloaded checkpoint is missing token embeddings")
    full_state.setdefault("lm_head.weight", full_state[embedding_key])

    expected_projection_keys = {
        f"model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(config.num_hidden_layers)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    expected_projection_keys.update(
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(config.num_hidden_layers)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    missing = expected_projection_keys - set(full_state)
    if missing:
        raise RuntimeError(
            f"Downloaded checkpoint is missing {len(missing)} projection tensors"
        )

    model_dir = pretrained_dir / "model"
    model_dir.mkdir(parents=True)
    for tp_rank in range(args.tp_size):
        shard = shard_state_dict(full_state, args.tp_size, tp_rank)
        output_file = (
            model_dir
            / f"dp_rank_00_tp_rank_{tp_rank:02d}_pp_rank_00.pt"
        )
        torch.save(shard, output_file)
        print(f"Saved {output_file}")
    print(f"QWEN3_TP_CHECKPOINT_OK: {output_root}")


if __name__ == "__main__":
    main()
