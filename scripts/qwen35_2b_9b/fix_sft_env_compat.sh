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
    python -m pip install --upgrade "pip" "setuptools" "wheel"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" \
    python -m pip install --upgrade --no-deps "trl==0.9.6"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" \
    python -m pip install --upgrade "jieba" "nltk" "huggingface_hub[cli]" "safetensors" "tokenizers"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" \
    python -m pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" python - <<'PY'
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
transformers_init = site_packages / "transformers" / "__init__.py"
alias_marker = "# qwen35_sft_compat_autovision2seq"
if transformers_init.exists():
    init_text = transformers_init.read_text()
    if alias_marker not in init_text:
        init_text += r'''

# qwen35_sft_compat_autovision2seq
try:
    AutoModelForVision2Seq
except NameError:
    try:
        AutoModelForVision2Seq = AutoModelForImageTextToText
    except NameError:
        try:
            AutoModelForVision2Seq = AutoModelForConditionalGeneration
        except NameError:
            try:
                AutoModelForVision2Seq = AutoModelForCausalLM
            except NameError:
                pass
'''
        transformers_init.write_text(init_text)
        print(f"patched {transformers_init}")

loader = site_packages / "llamafactory" / "model" / "loader.py"
old_import = "from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer"
new_import = """from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        try:
            from transformers import AutoModelForConditionalGeneration as AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = AutoModelForCausalLM"""
if loader.exists():
    loader_text = loader.read_text()
    if old_import in loader_text:
        loader.write_text(loader_text.replace(old_import, new_import))
        print(f"patched {loader}")

shim = site_packages / "sitecustomize.py"
body = r'''
"""Compatibility shims for older LLaMA-Factory imports.

LLaMA-Factory versions that import LongLoRA support may import private
Transformers LLaMA classes removed in newer Transformers. Qwen SFT does not use
these classes, but the import happens at package import time.
"""
try:
    import transformers as _tf

    if not hasattr(_tf, "AutoModelForVision2Seq"):
        for _name in (
            "AutoModelForImageTextToText",
            "AutoModelForConditionalGeneration",
            "AutoModelForCausalLM",
        ):
            if hasattr(_tf, _name):
                _tf.AutoModelForVision2Seq = getattr(_tf, _name)
                break
except Exception:
    pass

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
from transformers import Qwen3_5ForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import llamafactory
from llamafactory.model.loader import AutoModelForVision2Seq
print("transformers", transformers.__version__)
print("qwen3_5 class", Qwen3_5ForCausalLM.__name__)
print("vision fallback", AutoModelForVision2Seq.__name__)
print("imports ok")
PY
