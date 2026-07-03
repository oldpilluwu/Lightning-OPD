#!/usr/bin/env python3

from pathlib import Path
import site


def patch_file(path: Path) -> bool:
    if not path.exists():
        return False

    text = path.read_text()
    marker = "qwen35_text_only_ignore_visual"
    if marker in text:
        print(f"already patched {path}")
        return True

    old = """                if weights_not_loaded:
                    raise ValueError("Following weights were not initialized from "
                                     f"checkpoint: {weights_not_loaded}")
"""
    new = """                if weights_not_loaded:
                    # qwen35_text_only_ignore_visual: vLLM currently instantiates
                    # Qwen3.5 text-only checkpoints with a multimodal class. Text
                    # rollouts do not use the visual encoder, so ignore missing
                    # visual.* tensors while keeping strict checks for text weights.
                    weights_not_loaded = {
                        name for name in weights_not_loaded
                        if not name.startswith("visual.")
                    }
                if weights_not_loaded:
                    raise ValueError("Following weights were not initialized from "
                                     f"checkpoint: {weights_not_loaded}")
"""

    if old not in text:
        raise SystemExit(f"Could not find strict weight check block in {path}")

    path.write_text(text.replace(old, new))
    compile(path.read_text(), str(path), "exec")
    print(f"patched {path}")
    return True


def main() -> None:
    patched = False
    for site_packages in site.getsitepackages():
        root = Path(site_packages)
        candidates = [
            root / "vllm" / "model_executor" / "model_loader" / "default_loader.py",
        ]
        for candidate in candidates:
            patched = patch_file(candidate) or patched

    if not patched:
        raise SystemExit("No vLLM default_loader.py found to patch")


if __name__ == "__main__":
    main()
