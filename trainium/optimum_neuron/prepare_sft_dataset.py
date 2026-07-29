#!/usr/bin/env python3
"""Pre-tokenize Lightning-OPD SFT data exactly like LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

from llamafactory_compat import (
    LLAMAFACTORY_PREPROCESSING_BATCH_SIZE,
    LLAMAFACTORY_REFERENCE_COMMIT,
    Qwen3LlamaFactoryPackedProcessor,
)


def parquet_files(dataset_path: str) -> list[str]:
    path = Path(dataset_path).expanduser().resolve()
    if path.is_file() and path.suffix == ".parquet":
        return [str(path)]
    if path.is_dir():
        files = sorted(str(item) for item in path.rglob("*.parquet"))
        if files:
            return files
    raise ValueError(f"No Parquet files found at {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--expected-rows", type=int, default=300_000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug-only source-row limit, applied after expected-row validation.",
    )
    parser.add_argument("--preprocessing-num-workers", type=int, default=16)
    parser.add_argument(
        "--preprocessing-batch-size",
        type=int,
        default=LLAMAFACTORY_PREPROCESSING_BATCH_SIZE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory and bypass map caches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)

    files = parquet_files(args.dataset_path)
    source = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
    )
    if "messages" not in source.column_names:
        raise ValueError(
            f"Dataset requires a messages column; found {source.column_names}"
        )
    source_rows = len(source)
    if args.expected_rows and source_rows != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} rows, found {source_rows} "
            f"across {len(files)} Parquet file(s)"
        )
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        source = source.select(range(min(args.max_samples, source_rows)))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=False,
    )
    processor = Qwen3LlamaFactoryPackedProcessor(
        tokenizer=tokenizer,
        cutoff_len=args.cutoff_len,
    )
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "position_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
        }
    )
    tokenized = source.map(
        processor,
        batched=True,
        batch_size=args.preprocessing_batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=source.column_names,
        load_from_cache_file=not args.overwrite,
        features=features,
        desc="LLaMA-Factory-compatible Qwen3 SFT preprocessing",
    )
    if len(tokenized) == 0:
        raise RuntimeError(
            "Preprocessing produced zero packed rows. Check the messages schema and roles."
        )
    tokenized.save_to_disk(str(output_dir))
    tokenizer.save_pretrained(output_dir / "tokenizer")

    metadata = {
        "format": "lightning-opd-llamafactory-qwen3-packed-v1",
        "llamafactory_reference_commit": LLAMAFACTORY_REFERENCE_COMMIT,
        "model_id": args.model_id,
        "source_files": files,
        "source_rows": source_rows,
        "selected_source_rows": len(source),
        "packed_rows": len(tokenized),
        "cutoff_len": args.cutoff_len,
        "packing_capacity": args.cutoff_len - 1,
        "packing": True,
        "neat_packing": False,
        "train_on_prompt": False,
        "mask_history": False,
        "enable_thinking": True,
        "preprocessing_num_workers": args.preprocessing_num_workers,
        "preprocessing_batch_size": args.preprocessing_batch_size,
    }
    (output_dir / "reproduction_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
