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
    parser.add_argument("--storage-path", type=Path, default=Path("/root/models"))
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
    if _gib(meminfo["SwapTotal"]) < 32:
        warnings.append("Configure at least 32 GiB of NVMe-backed swap before the full run")

    disk = shutil.disk_usage(args.storage_path)
    print(f"Free storage at {args.storage_path}: {_gib(disk.free):.1f} GiB")
    if _gib(disk.free) < 150:
        failures.append("At least 150 GiB free storage is required")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
