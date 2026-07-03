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

    needle = "weights_not_loaded = weights_to_load - loaded_weights"
    lines = text.splitlines()
    out = []
    patched_count = 0

    for idx, line in enumerate(lines):
        out.append(line)
        if needle not in line:
            continue

        next_lines = "\n".join(lines[idx + 1 : idx + 8])
        if marker in next_lines:
            continue

        indent = line[: len(line) - len(line.lstrip())]
        out.extend(
            [
                f"{indent}# {marker}: vLLM currently instantiates Qwen3.5",
                f"{indent}# text-only checkpoints with a multimodal class. Text",
                f"{indent}# rollouts do not use visual encoder weights.",
                f"{indent}weights_not_loaded = {{",
                f"{indent}    name for name in weights_not_loaded",
                f'{indent}    if not name.startswith("visual.")',
                f"{indent}}}",
            ]
        )
        patched_count += 1

    if patched_count == 0:
        raise SystemExit(f"Could not find weight tracking assignment in {path}")

    path.write_text("\n".join(out) + "\n")
    compile(path.read_text(), str(path), "exec")
    print(f"patched {path} ({patched_count} insertion(s))")
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
