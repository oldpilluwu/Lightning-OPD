# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge Arrow IPC files produced by data_curation/pipeline.py into a single parquet.

After multi-GPU data generation, each worker writes Arrow files into
rank-specific subdirectories. This script merges them into one parquet
file for downstream consumption (SFT training or Lightning OPD preparation).

Usage:
    python data_curation/merge.py \
        --input-dir data/sft_data \
        --output data/sft_data/merged.parquet

    # With filtering: only keep samples with token count <= 16384
    python data_curation/merge.py \
        --input-dir data/sft_data \
        --output data/sft_data/merged.parquet \
        --max-tokens 16384
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Arrow IPC files into a single parquet file."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing Arrow files (searched recursively).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output parquet file path.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="If set, discard rows with tokens > this value.",
    )
    return parser.parse_args()


def merge_arrow_files(input_dir: str, output: str, max_tokens: int | None = None):
    input_path = Path(input_dir)
    arrow_files = sorted(input_path.rglob("*.arrow"))

    if not arrow_files:
        print(f"No Arrow files found in {input_dir}")
        return

    print(f"Found {len(arrow_files)} Arrow files in {input_dir}")

    tables = []
    total_rows = 0
    for f in tqdm(arrow_files, desc="Reading Arrow files"):
        with pa.OSFile(str(f), "rb") as source:
            table = ipc.open_file(source).read_all()
            tables.append(table)
            total_rows += len(table)

    merged = pa.concat_tables(tables)
    print(f"Total rows before filtering: {total_rows}")

    if max_tokens is not None and "tokens" in merged.column_names:
        tokens = merged.column("tokens").to_pylist()
        mask = [t <= max_tokens for t in tokens]
        merged = merged.filter(mask)
        filtered = total_rows - len(merged)
        print(f"Filtered {filtered} rows with tokens > {max_tokens}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = merged.to_pandas()
    df.to_parquet(output, index=False)

    print(f"Merged {len(df)} rows -> {output}")


if __name__ == "__main__":
    args = parse_args()
    merge_arrow_files(args.input_dir, args.output, args.max_tokens)
