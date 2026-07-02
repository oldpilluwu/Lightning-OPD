#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


WEIGHT_SUFFIXES = (
    ".bin",
    ".safetensors",
    ".pt",
)

WEIGHT_INDEX_NAMES = {
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
}

BASE_ALLOW_PATTERNS = [
    "*.json",
    "*.jinja",
    "*.model",
    "*.txt",
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]

BASE_IGNORE_PATTERNS = [
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.gguf",
    "*.onnx",
    "*.msgpack",
]


def copy_tree_files(src: Path, dst: Path, *, skip_weights: bool) -> None:
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if skip_weights and (path.suffix in WEIGHT_SUFFIXES or path.name in WEIGHT_INDEX_NAMES):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def copy_sft_weights(src: Path, dst: Path) -> int:
    count = 0
    for path in src.iterdir():
        if not path.is_file():
            continue
        if path.suffix in WEIGHT_SUFFIXES or path.name in WEIGHT_INDEX_NAMES:
            shutil.copy2(path, dst / path.name)
            count += 1
    return count


def read_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sft_checkpoint = Path(args.sft_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not (sft_checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Missing SFT config.json: {sft_checkpoint}")

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_snapshot = Path(
        snapshot_download(
            repo_id=args.base_model,
            allow_patterns=BASE_ALLOW_PATTERNS,
            ignore_patterns=BASE_IGNORE_PATTERNS,
        )
    )

    copy_tree_files(base_snapshot, output_dir, skip_weights=True)
    weights = copy_sft_weights(sft_checkpoint, output_dir)
    if weights == 0:
        raise FileNotFoundError(f"No model weight files found in {sft_checkpoint}")

    # Keep the fine-tuned generation/tokenizer files if the native SFT run wrote
    # any, but intentionally keep base config.json. vLLM currently expects the
    # base Qwen3.5 config wrapper, not Transformers 5's Qwen3_5TextConfig save.
    for name in ["generation_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
        src = sft_checkpoint / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    base_cfg = read_config(output_dir / "config.json")
    sft_cfg = read_config(sft_checkpoint / "config.json")
    report = {
        "base_model": args.base_model,
        "sft_checkpoint": str(sft_checkpoint),
        "vllm_checkpoint": str(output_dir),
        "base_model_type": base_cfg.get("model_type"),
        "base_architectures": base_cfg.get("architectures"),
        "sft_model_type": sft_cfg.get("model_type"),
        "sft_architectures": sft_cfg.get("architectures"),
        "weight_files": sorted(path.name for path in output_dir.iterdir() if path.suffix in WEIGHT_SUFFIXES),
    }
    (output_dir / "vllm_prepare_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
