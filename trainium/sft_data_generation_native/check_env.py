# SPDX-License-Identifier: Apache-2.0
"""Fail-fast environment check for the native TorchNeuron Beta-3 path."""

from __future__ import annotations

import importlib.metadata
import shutil

import torch

from trainium.sft_data_generation_native.pipeline import import_torchneuron


def version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "metadata unavailable"


def main() -> None:
    if shutil.which("neuron-ls") is None:
        raise SystemExit("neuron-ls is unavailable; run this inside the Beta-3 Neuron environment")
    backend_import = import_torchneuron()
    device = torch.device("neuron")
    compile_backends = [name for name in torch._dynamo.list_backends() if "neuron" in name.lower()]
    if not compile_backends:
        raise SystemExit("the installed PyTorch build exposes no Neuron torch.compile backend")
    print(f"torch={torch.__version__}")
    print(f"torch-neuronx={version('torch-neuronx')} import={backend_import}")
    print(f"neuronx-cc={version('neuronx-cc')}")
    print(f"device={device}")
    print(f"compile_backends={compile_backends}")


if __name__ == "__main__":
    main()
