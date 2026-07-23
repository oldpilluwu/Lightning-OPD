#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Split a JSONL file into balanced, contiguous shards with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split JSONL rows evenly into deterministic contiguous shards."
    )
    parser.add_argument("--input", required=True, help="Source JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Shard output directory.")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Fail unless the source contains exactly this many rows.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Output filename prefix, for example openthoughts3_300000.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_jsonl(
    input_path: str,
    output_dir: str,
    num_shards: int,
    prefix: str,
    expected_rows: int | None = None,
) -> dict:
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if expected_rows is not None and expected_rows < 0:
        raise ValueError("expected_rows cannot be negative")

    with source.open("rb") as handle:
        total_rows = sum(1 for _ in handle)
    if expected_rows is not None and total_rows != expected_rows:
        raise ValueError(f"{source}: found {total_rows} rows, expected {expected_rows}")
    if total_rows < num_shards:
        raise ValueError(
            f"cannot split {total_rows} rows into {num_shards} non-empty shards"
        )

    destination.mkdir(parents=True, exist_ok=True)
    width = 5
    shard_paths = [
        destination
        / f"{prefix}_node{index:0{width}d}-of-{num_shards:0{width}d}.jsonl"
        for index in range(num_shards)
    ]
    temporary_paths = [
        path.with_name(f".{path.name}.tmp.{os.getpid()}") for path in shard_paths
    ]
    base_size, remainder = divmod(total_rows, num_shards)
    shard_sizes = [
        base_size + (1 if index < remainder else 0)
        for index in range(num_shards)
    ]

    try:
        with source.open("rb") as input_handle, ExitStack() as stack:
            output_handles = [
                stack.enter_context(path.open("wb")) for path in temporary_paths
            ]
            shard_index = 0
            rows_in_shard = 0
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    raise ValueError(f"{source}:{line_number}: blank JSONL row")
                output_handles[shard_index].write(line)
                rows_in_shard += 1
                if (
                    rows_in_shard == shard_sizes[shard_index]
                    and shard_index + 1 < num_shards
                ):
                    shard_index += 1
                    rows_in_shard = 0

        for temporary_path, shard_path in zip(temporary_paths, shard_paths):
            os.replace(temporary_path, shard_path)
    except BaseException:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise

    source_hash = sha256(source)
    shards = []
    start = 0
    for index, (path, rows) in enumerate(zip(shard_paths, shard_sizes)):
        end = start + rows
        shards.append(
            {
                "index": index,
                "rows": rows,
                "start_row": start,
                "end_row_exclusive": end,
                "filename": path.name,
                "sha256": sha256(path),
            }
        )
        start = end

    manifest = {
        "source": str(source),
        "source_rows": total_rows,
        "source_sha256": source_hash,
        "num_shards": num_shards,
        "shards": shards,
    }
    manifest_path = destination / f"{prefix}_manifest.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{os.getpid()}"
    )
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_manifest, manifest_path)

    checksums_path = destination / f"{prefix}_SHA256SUMS"
    temporary_checksums = checksums_path.with_name(
        f".{checksums_path.name}.tmp.{os.getpid()}"
    )
    with temporary_checksums.open("w", encoding="utf-8", newline="\n") as handle:
        for shard in shards:
            handle.write(f"{shard['sha256']}  {shard['filename']}\n")
    os.replace(temporary_checksums, checksums_path)

    print(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksums_path}")
    return manifest


if __name__ == "__main__":
    arguments = parse_args()
    split_jsonl(
        arguments.input,
        arguments.output_dir,
        arguments.num_shards,
        arguments.prefix,
        arguments.expected_rows,
    )
