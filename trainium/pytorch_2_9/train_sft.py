#!/usr/bin/env python3
"""Direct PyTorch 2.9/XLA/NxD full-model SFT for Qwen3-4B-Base."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from datasets import load_from_disk
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen3Config, default_data_collator
from transformers.optimization import get_cosine_schedule_with_warmup

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.random import (
    model_parallel_xla_manual_seed,
)
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

from modeling_qwen3_nxd import Qwen3ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--tokenized-dataset", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--ddp-timeout", type=int, default=180_000_000)
    parser.add_argument("--resume-tag", default="")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="wandb")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def initialize_distributed(timeout_seconds: int) -> None:
    if dist.is_initialized():
        return
    timeout = timedelta(seconds=timeout_seconds)
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa: F401

        dist.init_process_group("xla", init_method="pjrt://", timeout=timeout)
    else:
        dist.init_process_group("xla", timeout=timeout)


def validate_reproduction_metadata(
    dataset_path: Path,
    model_id: str,
    sequence_length: int,
    smoke: bool,
) -> dict:
    metadata_path = dataset_path / "reproduction_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing preprocessing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "format": "lightning-opd-llamafactory-qwen3-packed-v1",
        "llamafactory_reference_commit": (
            "9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d"
        ),
        "model_id": model_id,
        "cutoff_len": sequence_length,
        "packing_capacity": sequence_length - 1,
        "packing": True,
        "neat_packing": False,
        "train_on_prompt": False,
        "mask_history": False,
        "enable_thinking": True,
        "overwrite_cache": True,
        "preprocessing_batch_size": 1000,
    }
    if not smoke:
        expected.update(
            {
                "source_rows": 300_000,
                "selected_source_rows": 300_000,
                "preprocessing_num_workers": 16,
            }
        )
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Preprocessing does not reproduce LLaMA-Factory: {mismatches}")
    return metadata


def create_dataloader(
    dataset_path: Path,
    sequence_length: int,
    micro_batch_size: int,
    data_seed: int,
    num_workers: int,
) -> tuple[DataLoader, DistributedSampler]:
    dataset = load_from_disk(str(dataset_path))
    required_columns = {"input_ids", "attention_mask", "position_ids", "labels"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Tokenized dataset is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError("Tokenized dataset has no packed rows")
    first_row = dataset[0]
    bad_lengths = {
        name: len(first_row[name])
        for name in required_columns
        if len(first_row[name]) != sequence_length
    }
    if bad_lengths:
        raise ValueError(f"First packed row has inconsistent tensor lengths: {bad_lengths}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=data_seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def build_model(config: Qwen3Config) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(config)


def write_trainer_state(
    output_dir: Path,
    global_step: int,
    log_history: list[dict[str, float | int]],
) -> None:
    if not xm.is_master_ordinal(local=False):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": global_step,
        "log_history": log_history,
        "backend": "pytorch-2.9-xla-nxd",
        "full_model": True,
        "lora": False,
    }
    (output_dir / "trainer_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def save_training_checkpoint(
    output_dir: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    user_content: dict,
    save_total_limit: int,
) -> None:
    nxd.save_checkpoint(
        output_dir,
        tag=tag,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        user_content=user_content,
        num_workers=8,
        use_xser=True,
        async_save=False,
        num_kept_ckpts=save_total_limit,
    )


@record
def main() -> None:
    args = parse_args()
    initialize_distributed(args.ddp_timeout)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.tokenized_dataset).expanduser().resolve()
    checkpoint_path = Path(args.pretrained_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if (
        not args.resume_tag
        and output_dir.is_dir()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"{output_dir} is not empty and overwrite_output_dir is false"
        )
    validate_reproduction_metadata(
        dataset_path, args.model_id, args.sequence_length, args.smoke
    )
    if not (checkpoint_path / "pretrained_weight" / "model").is_dir():
        raise FileNotFoundError(
            "Missing converted TP checkpoint at "
            f"{checkpoint_path / 'pretrained_weight' / 'model'}"
        )
    if args.sequence_length % 2048:
        raise ValueError("Neuron flash attention requires sequence length multiple of 2048")

    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=args.tensor_parallel_size,
        optimizer_config={
            "zero_one_enabled": False,
            "grad_clipping": True,
            "max_grad_norm": args.max_grad_norm,
        },
        sequence_parallel=True,
        # Match the original LLaMA-Factory run: gradient checkpointing disabled.
        activation_checkpoint_config=None,
        mixed_precision_config={
            # NxD 0.19 only supports these FP32 optimizer features with
            # ZeRO-1. Keep them disabled to reproduce the source run's
            # explicit ZeRO-0 configuration.
            "use_master_weights": False,
            "use_fp32_grad_acc": False,
            "use_master_weights_in_ckpt": False,
        },
    )
    # neuronx_distributed_config initializes the TP groups required by this
    # model-parallel RNG tracker.
    model_parallel_xla_manual_seed(args.seed)

    config = Qwen3Config.from_pretrained(args.model_id)
    config.use_cache = False
    config.torch_dtype = torch.bfloat16
    config.sequence_parallel_enabled = True
    model = nxd.initialize_parallel_model(
        nxd_config, build_model, True, config
    )

    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("All model parameters must be trainable for full SFT")
    if any("lora" in name.lower() for name, _ in model.named_parameters()):
        raise RuntimeError("LoRA parameters detected in full-model training")

    optimizer = nxd.initialize_parallel_optimizer(
        nxd_config,
        torch.optim.AdamW,
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    if args.resume_tag:
        user_content = nxd.load_checkpoint(
            str(output_dir),
            tag=args.resume_tag,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        global_step = int((user_content or {}).get("global_step", 0))
        epoch = int((user_content or {}).get("epoch", 0))
    else:
        nxd.load_checkpoint(
            str(checkpoint_path),
            tag="pretrained_weight",
            model=model,
            optimizer=None,
            scheduler=None,
        )
        global_step = 0
        epoch = 0

    device = torch_xla.device()
    loader, sampler = create_dataloader(
        dataset_path,
        args.sequence_length,
        args.micro_batch_size,
        args.data_seed,
        args.dataloader_workers,
    )
    device_loader = pl.MpDeviceLoader(loader, device)
    dp_size = parallel_state.get_data_parallel_size()
    global_batch = (
        dp_size * args.micro_batch_size * args.gradient_accumulation_steps
    )
    local_parameters = sum(parameter.numel() for parameter in model.parameters())

    xm.master_print("=== Direct PyTorch 2.9 Qwen3 full SFT ===")
    xm.master_print(f"Model:             {args.model_id}")
    xm.master_print(f"Dataset:           {dataset_path}")
    xm.master_print(f"TP / DP:           {args.tensor_parallel_size} / {dp_size}")
    xm.master_print(f"Sequence length:   {args.sequence_length}")
    xm.master_print(f"Global batch:      {global_batch}")
    xm.master_print(f"Local parameters:  {local_parameters:,}")
    xm.master_print("Trainable mode:    FULL MODEL (no PEFT/LoRA)")

    wandb_run = None
    if args.report_to == "wandb" and xm.is_master_ordinal(local=False):
        import wandb

        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "lightning-opd"),
            name="qwen3-4b-base-open-thoughts3-qwen3-8b",
            config=vars(args),
        )

    optimizer.zero_grad()
    accumulated_loss = torch.zeros(1, dtype=torch.float32, device=device)
    micro_step = 0
    last_loss = float("nan")
    log_history: list[dict[str, float | int]] = []
    start_time = time.time()

    while global_step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch in device_loader:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                labels=batch["labels"],
            )
            if output.loss is None:
                raise RuntimeError("Model did not return a training loss")
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.detach()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps:
                continue

            xm.mark_step()
            # Keep metrics out of the compiled optimizer graph. On the PyTorch
            # 2.9/NxD 0.19 Trn2 stack, materializing a standalone xm.all_reduce
            # over a DP subgroup can produce an invalid send/recv target. The
            # model loss is already identical across TP ranks because
            # parallel_cross_entropy performs its TP reduction, so averaging
            # the local scalar across the host mesh gives the same DP mean.
            local_loss = accumulated_loss.detach().clone()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            accumulated_loss.zero_()

            def log_step(loss_tensor: torch.Tensor, step: int) -> None:
                nonlocal last_loss
                local_loss_value = float(loss_tensor.cpu().item())
                last_loss = float(
                    xm.mesh_reduce(
                        f"train-loss-{step}",
                        local_loss_value,
                        lambda values: sum(values) / len(values),
                    )
                )
                if xm.is_master_ordinal(local=False):
                    elapsed = time.time() - start_time
                    lr = optimizer.param_groups[0]["lr"]
                    log_history.append(
                        {
                            "step": step,
                            "loss": last_loss,
                            "learning_rate": lr,
                        }
                    )
                    print(
                        f"step={step} loss={last_loss:.6f} "
                        f"lr={lr:.8g} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {"train/loss": last_loss, "train/learning_rate": lr},
                            step=step,
                        )

            if global_step % args.logging_steps == 0:
                xm.add_step_closure(log_step, (local_loss, global_step))

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                xm.add_step_closure(
                    save_training_checkpoint,
                    (
                        str(output_dir),
                        f"step_{global_step}",
                        model,
                        optimizer,
                        scheduler,
                        {"epoch": epoch, "global_step": global_step},
                        args.save_total_limit,
                    ),
                )

            if global_step >= args.max_steps:
                xm.mark_step()
                break
        epoch += 1

    xm.rendezvous("training-complete")
    write_trainer_state(
        output_dir,
        global_step,
        log_history,
    )
    xm.rendezvous("trainer-state-written")
    if wandb_run is not None:
        wandb_run.finish()
    xm.master_print("TRAINING_OK")


if __name__ == "__main__":
    main()
