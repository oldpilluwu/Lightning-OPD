#!/usr/bin/env python3
"""Small native TorchNeuron forward/backward/AdamW environment test."""

from __future__ import annotations

import math
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx


class TinyQwenBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.attention_norm = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.gate = nn.Linear(dim, 4 * dim, bias=False)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, dim = hidden.shape
        normalized = self.attention_norm(hidden)
        qkv = self.qkv(normalized).view(
            batch,
            sequence,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, dim)
        hidden = hidden + self.output(attended)
        normalized = self.ffn_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class TinyQwenLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.block = TinyQwenBlock(dim, heads)
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.block(self.embedding(input_ids))
        return self.output(self.norm(hidden))


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("neuron")
    vocab_size = 1024
    model = TinyQwenLM(vocab_size=vocab_size, dim=256, heads=8)
    model = model.to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    input_ids = torch.randint(0, vocab_size, (4, 128), device=device)
    labels = torch.randint(0, vocab_size, (4, 128), device=device)
    losses: list[float] = []
    for step in range(5):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(),
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite native training loss at step {step}: {loss_value}")
        losses.append(loss_value)
        print(f"step={step} loss={loss_value:.6f}", flush=True)

    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"Native training loss did not decrease: first={losses[0]}, last={losses[-1]}"
        )
    fallback_ops = torch_neuronx.get_fallback_ops()
    print(f"fallback_ops={fallback_ops}")
    if fallback_ops:
        raise RuntimeError(f"Native training smoke used CPU fallback ops: {fallback_ops}")
    print("NATIVE_TRAIN_SMOKE_OK")


if __name__ == "__main__":
    main()
