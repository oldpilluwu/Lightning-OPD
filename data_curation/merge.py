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
import os
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
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
        raise FileNotFoundError(f"No Arrow files found in {input_dir}")

    print(f"Found {len(arrow_files)} Arrow files in {input_dir}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    writer = None
    schema = None
    total_rows = 0
    written_rows = 0
    try:
        for arrow_file in tqdm(arrow_files, desc="Merging Arrow files"):
            with arrow_file.open("rb") as source:
                table = ipc.open_file(source).read_all()
            total_rows += len(table)

            if max_tokens is not None and "tokens" in table.column_names:
                table = table.filter(pc.less_equal(table["tokens"], max_tokens))

            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(temporary_path, schema, compression="snappy")
            elif table.schema != schema:
                raise ValueError(f"{arrow_file}: schema does not match earlier shards")

            writer.write_table(table)
            written_rows += len(table)

        assert writer is not None
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    filtered = total_rows - written_rows
    print(f"Total rows before filtering: {total_rows}")
    if max_tokens is not None:
        print(f"Filtered {filtered} rows with tokens > {max_tokens}")
    print(f"Merged {written_rows} rows -> {output}")


if __name__ == "__main__":
    args = parse_args()
    merge_arrow_files(args.input_dir, args.output, args.max_tokens)
