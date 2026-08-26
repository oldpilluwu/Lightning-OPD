#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Read-only preflight checks for the single-A6000 OPD experiment."""

import argparse
import shutil
import subprocess
from pathlib import Path


def _gib(value: int) -> float:
    return value / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path.cwd() / "models",
        help="Artifact directory (default: ./models). It need not exist yet.",
    )
    parser.add_argument(
        "--minimum-free-storage-gib",
        type=float,
        default=120,
        help="Minimum required free storage in GiB (default: 120).",
    )
    args = parser.parse_args()

    failures = []
    warnings = []
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()
        if len(output) != 1:
            failures.append(f"Expected exactly one visible GPU, found {len(output)}")
        else:
            name, memory_mib = (item.strip() for item in output[0].rsplit(",", 1))
            print(f"GPU: {name}, {int(memory_mib) / 1024:.1f} GiB")
            if int(memory_mib) < 47_000:
                failures.append("At least 47,000 MiB of visible GPU memory is required")
    except (OSError, subprocess.CalledProcessError) as error:
        failures.append(f"Unable to query nvidia-smi: {error}")

    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    print(f"Host RAM: {_gib(meminfo['MemTotal']):.1f} GiB")
    print(f"Swap: {_gib(meminfo['SwapTotal']):.1f} GiB")
    if _gib(meminfo["SwapTotal"]) < 16:
        warnings.append("Configure at least 16 GiB of NVMe-backed swap before the full run")

    storage_path = args.storage_path.expanduser().absolute()
    probe_path = storage_path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        disk = shutil.disk_usage(probe_path)
        location = str(storage_path)
        if probe_path != storage_path:
            location += f" (filesystem checked at {probe_path})"
        print(f"Free storage for {location}: {_gib(disk.free):.1f} GiB")
        if _gib(disk.free) < args.minimum_free_storage_gib:
            failures.append(
                f"At least {args.minimum_free_storage_gib:g} GiB free storage is required"
            )
    except OSError as error:
        failures.append(f"Unable to inspect storage path {storage_path}: {error}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
