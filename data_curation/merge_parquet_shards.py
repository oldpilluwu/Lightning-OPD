#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Stream multiple curation parquet shards into one validated parquet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge independently generated parquet shards without loading them all into RAM."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input parquet shards, in the desired output order.",
    )
    parser.add_argument("--output", required=True, help="Final parquet path.")
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Fail unless the merged parquet has exactly this many rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Rows read per streaming batch (default: 256).",
    )
    return parser.parse_args()


def merge_parquet_shards(
    inputs: list[str],
    output: str,
    expected_rows: int | None = None,
    batch_size: int = 256,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if expected_rows is not None and expected_rows < 0:
        raise ValueError("expected_rows cannot be negative")

    input_paths = [Path(value).resolve() for value in inputs]
    output_path = Path(output).resolve()
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("input parquet paths must be unique")
    if output_path in input_paths:
        raise ValueError("output must not overwrite an input shard")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total_rows = 0

    try:
        for path in input_paths:
            parquet = pq.ParquetFile(path)
            current_schema = parquet.schema_arrow
            missing = {"messages", "tokens"} - set(current_schema.names)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            if schema is None:
                schema = current_schema
                writer = pq.ParquetWriter(temporary_path, schema, compression="snappy")
            elif current_schema != schema:
                raise ValueError(f"{path}: schema does not match the first shard")

            for batch in parquet.iter_batches(batch_size=batch_size):
                assert writer is not None
                writer.write_batch(batch)
                total_rows += batch.num_rows

        if writer is None:
            raise ValueError("at least one input parquet is required")
        writer.close()
        writer = None

        if expected_rows is not None and total_rows != expected_rows:
            raise ValueError(
                f"merged row count is {total_rows}, expected {expected_rows}"
            )

        os.replace(temporary_path, output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    merged = pq.ParquetFile(output_path)
    if merged.metadata.num_rows != total_rows:
        raise RuntimeError(
            f"{output_path}: wrote {merged.metadata.num_rows} rows, expected {total_rows}"
        )
    print(f"Merged {len(input_paths)} shards and {total_rows} rows -> {output_path}")
    return total_rows


if __name__ == "__main__":
    arguments = parse_args()
    merge_parquet_shards(
        arguments.inputs,
        arguments.output,
        arguments.expected_rows,
        arguments.batch_size,
    )
