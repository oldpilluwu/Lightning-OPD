#!/usr/bin/env python3
"""Full-parameter Qwen3 SFT on Trainium with Optimum Neuron."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, HfArgumentParser, default_data_collator

from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

from llamafactory_compat import fix_qwen3_tokenizer


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="Qwen/Qwen3-4B-Base",
        metadata={"help": "Hugging Face model ID or a local full-model checkpoint."},
    )
    tokenized_dataset: str = field(
        default="",
        metadata={"help": "Dataset created by prepare_sft_dataset.py."},
    )
    sequence_length: int = field(
        default=16384,
        metadata={"help": "Expected fixed packed sequence length."},
    )


def load_reproduction_dataset(
    path_value: str,
    sequence_length: int,
    model_id: str,
):
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Tokenized dataset directory does not exist: {path}")
    metadata_path = path / "reproduction_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"{path} was not created by prepare_sft_dataset.py "
            "(missing reproduction_metadata.json)"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "lightning-opd-llamafactory-qwen3-packed-v1":
        raise ValueError(f"Unsupported tokenized dataset format in {metadata_path}")
    if metadata.get("cutoff_len") != sequence_length:
        raise ValueError(
            f"Dataset sequence length is {metadata.get('cutoff_len')}, "
            f"but --sequence_length is {sequence_length}"
        )
    required_metadata = {
        "model_id": model_id,
        "packing": True,
        "neat_packing": False,
        "train_on_prompt": False,
        "mask_history": False,
        "enable_thinking": True,
        "preprocessing_batch_size": 1000,
    }
    mismatched_metadata = {
        key: (metadata.get(key), expected)
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    if sequence_length == 16384:
        production_metadata = {
            "source_rows": 300_000,
            "selected_source_rows": 300_000,
            "preprocessing_num_workers": 16,
        }
        mismatched_metadata.update(
            {
                key: (metadata.get(key), expected)
                for key, expected in production_metadata.items()
                if metadata.get(key) != expected
            }
        )
    if mismatched_metadata:
        raise ValueError(
            "Tokenized data does not reproduce the original SFT preprocessing: "
            f"{mismatched_metadata}"
        )

    dataset = load_from_disk(str(path))
    required = {"input_ids", "attention_mask", "position_ids", "labels"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Tokenized dataset is missing columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError("Tokenized dataset has no packed samples")
    first = dataset[0]
    bad_lengths = {
        key: len(first[key])
        for key in required
        if len(first[key]) != sequence_length
    }
    if bad_lengths:
        raise ValueError(
            f"Packed tensor lengths must all be {sequence_length}: {bad_lengths}"
        )
    return dataset, metadata


def save_loss_plot(log_history: list[dict], output_dir: str) -> None:
    if os.environ.get("RANK", "0") != "0":
        return
    points = [
        (entry["step"], entry["loss"])
        for entry in log_history
        if "step" in entry and "loss" in entry
    ]
    if not points:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping trainer_loss.png")
        return
    steps, losses = zip(*points)
    figure, axis = plt.subplots()
    axis.plot(steps, losses)
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title("Qwen3-4B full SFT loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "trainer_loss.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if not script_args.tokenized_dataset:
        raise ValueError("--tokenized_dataset is required")

    train_dataset, metadata = load_reproduction_dataset(
        script_args.tokenized_dataset,
        script_args.sequence_length,
        script_args.model_id,
    )
    tokenizer_path = (
        Path(script_args.tokenized_dataset).expanduser().resolve() / "tokenizer"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path.is_dir() else script_args.model_id,
        trust_remote_code=False,
    )
    fix_qwen3_tokenizer(tokenizer)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        script_args.model_id,
        training_args.trn_config,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "Full SFT requires every model parameter to be trainable; "
            f"found {trainable_parameters:,}/{total_parameters:,}"
        )
    if os.environ.get("RANK", "0") == "0":
        print(f"Full-model trainable parameters: {trainable_parameters:,}")
        print(
            "LLaMA-Factory-compatible packed rows: "
            f"{metadata['packed_rows']:,} at length {metadata['cutoff_len']:,}"
        )

    trainer = NeuronTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    save_loss_plot(trainer.state.log_history, training_args.output_dir)


if __name__ == "__main__":
    main()
