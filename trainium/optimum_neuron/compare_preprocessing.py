#!/usr/bin/env python3
"""Byte-for-byte comparison with a checked-out LLaMA-Factory processor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from llamafactory_compat import (
    LLAMAFACTORY_REFERENCE_COMMIT,
    Qwen3LlamaFactoryPackedProcessor,
    normalize_messages,
)
from prepare_sft_dataset import parquet_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamafactory-repo", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--num-samples", type=int, default=1000)
    return parser.parse_args()


def aligned_batch(raw_messages: list) -> dict[str, list]:
    batch = {
        "_prompt": [],
        "_response": [],
        "_system": [],
        "_tools": [],
        "_images": [],
        "_videos": [],
        "_audios": [],
    }
    for raw in raw_messages:
        system, messages = normalize_messages(raw)
        batch["_prompt"].append(messages[:-1])
        batch["_response"].append(messages[-1:])
        batch["_system"].append(system)
        batch["_tools"].append("")
        batch["_images"].append(None)
        batch["_videos"].append(None)
        batch["_audios"].append(None)
    return batch


def main() -> None:
    args = parse_args()
    repo = Path(args.llamafactory_repo).expanduser().resolve()
    source_dir = repo / "src"
    if not source_dir.is_dir():
        raise ValueError(f"Not a LLaMA-Factory checkout: {repo}")
    sys.path.insert(0, str(source_dir))

    from llamafactory.data.processor.supervised import (  # noqa: PLC0415
        PackedSupervisedDatasetProcessor,
    )
    from llamafactory.data.template import (  # noqa: PLC0415
        get_template_and_fix_tokenizer,
    )
    from llamafactory.hparams import DataArguments  # noqa: PLC0415

    source = load_dataset(
        "parquet",
        data_files={"train": parquet_files(args.dataset_path)},
        split=f"train[:{args.num_samples}]",
    )
    raw_messages = source["messages"]

    ours_tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ours = Qwen3LlamaFactoryPackedProcessor(
        tokenizer=ours_tokenizer,
        cutoff_len=args.cutoff_len,
    )({"messages": raw_messages})

    reference_tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    data_args = DataArguments(
        template="qwen3",
        cutoff_len=args.cutoff_len,
        packing=True,
        neat_packing=False,
        train_on_prompt=False,
        mask_history=False,
        enable_thinking=True,
    )
    template = get_template_and_fix_tokenizer(reference_tokenizer, data_args)
    reference_processor = PackedSupervisedDatasetProcessor(
        template=template,
        tokenizer=reference_tokenizer,
        processor=None,
        data_args=data_args,
    )
    reference = reference_processor.preprocess_dataset(
        aligned_batch(raw_messages)
    )

    compared_keys = ("input_ids", "attention_mask", "position_ids", "labels")
    for key in compared_keys:
        reference_value = list(reference[key])
        if ours[key] != reference_value:
            raise AssertionError(
                f"{key} differs: ours has {len(ours[key])} packs, "
                f"LLaMA-Factory has {len(reference_value)}"
            )
    print(
        f"PASS: {args.num_samples} source rows produced "
        f"{len(ours['input_ids'])} byte-for-byte equivalent packs"
    )
    print(f"Audited LLaMA-Factory commit: {LLAMAFACTORY_REFERENCE_COMMIT}")


if __name__ == "__main__":
    main()
