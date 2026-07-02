#!/usr/bin/env bash

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
    if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        CONDA_BIN="${HOME}/miniconda3/bin/conda"
    else
        echo "conda not found. Set CONDA_BIN=/path/to/conda." >&2
        exit 1
    fi
fi

SFT_ENV="${SFT_ENV:-qwen35-sft}"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" \
    python -m pip install --force-reinstall "trl==0.9.6" "transformers>=4.57.0,<5.0.0" "jieba" "nltk"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" python - <<'PY'
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
shim = site_packages / "sitecustomize.py"
body = r'''
"""Compatibility shims for older LLaMA-Factory imports.

LLaMA-Factory versions that import LongLoRA support may import private
Transformers LLaMA classes removed in newer Transformers. Qwen SFT does not use
these classes, but the import happens at package import time.
"""
try:
    import transformers.models.llama.modeling_llama as _llama

    if not hasattr(_llama, "LlamaFlashAttention2") and hasattr(_llama, "LlamaAttention"):
        _llama.LlamaFlashAttention2 = _llama.LlamaAttention
    if not hasattr(_llama, "LlamaSdpaAttention") and hasattr(_llama, "LlamaAttention"):
        _llama.LlamaSdpaAttention = _llama.LlamaAttention
except Exception:
    pass

try:
    import importlib.util
    import transformers.utils as _tf_utils

    if not hasattr(_tf_utils, "is_jieba_available"):
        def is_jieba_available():
            return importlib.util.find_spec("jieba") is not None
        _tf_utils.is_jieba_available = is_jieba_available

    if not hasattr(_tf_utils, "is_nltk_available"):
        def is_nltk_available():
            return importlib.util.find_spec("nltk") is not None
        _tf_utils.is_nltk_available = is_nltk_available
except Exception:
    pass
'''
shim.write_text(body)
print(f"wrote {shim}")
PY

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" python - <<'PY'
import transformers
from transformers import AutoModelForVision2Seq
from trl import AutoModelForCausalLMWithValueHead
import llamafactory
print("transformers", transformers.__version__)
print("imports ok")
PY
