#!/usr/bin/env python3
"""Register the Lightning-OPD SFT dataloader and run TorchTitan natively."""

from __future__ import annotations

import math
import os

# The native backend must register the ``neuron`` device before TorchTitan
# imports and caches PyTorch's available device type.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_neuronx

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.models.qwen3 import get_train_spec as get_qwen3_train_spec
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec
from torchtitan.train import Trainer, main

from trainium.native_torchtitan.sft_data import build_sft_dataloader


def _build_llamafactory_cosine_lr_schedulers(
    optimizers,
    lr_scheduler_config,
    training_steps: int,
) -> LRSchedulersContainer:
    """Match Transformers' cosine schedule used by LlamaFactory."""
    warmup_steps = int(lr_scheduler_config.warmup_steps)
    if warmup_steps < 0 or warmup_steps > training_steps:
        raise ValueError("warmup_steps must be between zero and training_steps")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, training_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LRSchedulersContainer(optimizers, lr_lambda)


def _register_sft_train_spec() -> None:
    qwen3_spec = get_qwen3_train_spec()
    register_train_spec(
        "qwen3_sft",
        TrainSpec(
            model_cls=qwen3_spec.model_cls,
            model_args=qwen3_spec.model_args,
            parallelize_fn=qwen3_spec.parallelize_fn,
            pipelining_fn=qwen3_spec.pipelining_fn,
            build_optimizers_fn=qwen3_spec.build_optimizers_fn,
            build_lr_schedulers_fn=_build_llamafactory_cosine_lr_schedulers,
            build_dataloader_fn=build_sft_dataloader,
            build_tokenizer_fn=qwen3_spec.build_tokenizer_fn,
            build_loss_fn=qwen3_spec.build_loss_fn,
            build_validator_fn=None,
            build_metrics_processor_fn=qwen3_spec.build_metrics_processor_fn,
            state_dict_adapter=qwen3_spec.state_dict_adapter,
        ),
    )


if __name__ == "__main__":
    if torch.device("neuron").type != "neuron":
        raise RuntimeError("TorchNeuron did not register the native neuron device")
    registered_backends = getattr(torch.distributed.Backend, "backend_list", ())
    if "neuron" not in registered_backends:
        raise RuntimeError("TorchNeuron did not register the neuron process-group backend")
    _register_sft_train_spec()
    main(Trainer)
