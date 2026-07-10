#!/usr/bin/env python3
"""Locate DSpark's compatibility boundary in a vLLM-Neuron stack."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def import_probe(module_name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        return False, f"parent package unavailable: {exc}"
    if spec is None:
        return False, "not found"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # A probe must report optional backend import errors.
        return False, f"found but import failed: {type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__file__", spec.origin))


def package_mentions(package_name: str, needle: str) -> list[str]:
    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    hits: list[str] = []
    for root in spec.submodule_search_locations:
        for path in Path(root).rglob("*.py"):
            try:
                if needle.lower() in path.read_text(errors="ignore").lower():
                    hits.append(str(path))
            except OSError:
                continue
    return hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--speculator-model", default="deepseek-ai/dspark_qwen3_8b_block7"
    )
    parser.add_argument(
        "--speculative-method",
        choices=("dspark", "draft_model"),
        default="dspark",
        help=(
            "dspark requests the faithful upstream algorithm; draft_model is "
            "only a diagnostic of NxDI generic fused speculation"
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--attempt-engine", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=== Runtime ===")
    print(f"host={platform.platform()}")
    print(f"python={sys.version.split()[0]} ({sys.executable})")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    for package in (
        "torch",
        "torch-neuronx",
        "neuronx-distributed",
        "neuronx-distributed-inference",
        "vllm",
        "vllm-neuron",
    ):
        print(f"{package}={distribution_version(package)}")

    probes = (
        "torch_neuronx",
        "neuronx_distributed_inference",
        "vllm_neuron",
        "vllm.model_executor.models.qwen3_dspark",
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator",
    )
    print("\n=== Import probes ===")
    results: dict[str, bool] = {}
    for module_name in probes:
        ok, detail = import_probe(module_name)
        results[module_name] = ok
        print(f"{'PASS' if ok else 'FAIL'} {module_name}: {detail}")

    print("\n=== Source capability scan ===")
    neuron_hits = package_mentions("vllm_neuron", "dspark")
    print(f"vllm-neuron DSpark references: {len(neuron_hits)}")
    for path in neuron_hits[:20]:
        print(f"  {path}")
    upstream_hits = package_mentions("vllm", "dspark")
    print(f"vLLM DSpark references: {len(upstream_hits)}")
    for path in upstream_hits[:20]:
        print(f"  {path}")

    model_module = "vllm.model_executor.models.qwen3_dspark"
    gpu_worker_module = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
    if not results.get(model_module):
        print("\nRESULT: this vLLM revision does not contain the Qwen3 DSpark model.")
        if not args.attempt_engine or args.speculative_method == "dspark":
            return 20
    if not neuron_hits:
        if results.get(model_module):
            print(
                "\nRESULT: vLLM contains DSpark, but the installed vllm-neuron "
                "plugin contains no DSpark integration."
            )
        else:
            print(
                "\nRESULT: neither installed vLLM nor the vllm-neuron plugin "
                "contains a DSpark integration."
            )
        print(
            "The implementation is under vLLM's GPU worker; importing that "
            f"module {'succeeded' if results.get(gpu_worker_module) else 'failed'}."
        )
        if not args.attempt_engine:
            return 21

    if not args.attempt_engine:
        print("\nRESULT: static compatibility checks passed; engine was not requested.")
        return 0

    print("\n=== Neuron engine construction ===")
    print("This may download both checkpoints and trigger Neuron compilation.")
    if args.speculative_method == "draft_model":
        print(
            "WARNING: draft_model is not DSpark. It tests whether NxDI can load "
            "the checkpoint as an ordinary causal draft and does not execute "
            "DSpark's non-causal block proposal or Markov head."
        )
    from vllm import LLM

    override_neuron_config = {
        "max_context_length": args.max_model_len,
        "seq_len": args.max_model_len,
        "ctx_batch_size": 1,
        "batch_size": args.max_num_seqs,
        "enable_bucketing": False,
    }
    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        # With prefix caching disabled, vllm-neuron uses one contiguous cache
        # allocation per compiled sequence slot.
        num_gpu_blocks_override=args.max_num_seqs,
        enable_prefix_caching=False,
        trust_remote_code=True,
        additional_config={"override_neuron_config": override_neuron_config},
        speculative_config={
            "method": args.speculative_method,
            "model": args.speculator_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    )
    output = llm.generate(["The capital of France is"], use_tqdm=False)
    print(output[0].outputs[0].text)
    if args.speculative_method == "dspark":
        print("\nRESULT: DSpark generated successfully through the Neuron backend.")
    else:
        print("\nRESULT: the DSpark checkpoint loaded as a generic NxDI draft model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
