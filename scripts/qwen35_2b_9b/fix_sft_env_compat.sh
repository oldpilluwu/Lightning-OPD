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
    python -m pip install --upgrade "jieba" "nltk" "huggingface_hub[cli]" "safetensors" "tokenizers>=0.22.0,<=0.23.0"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" \
    python -m pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"

"${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" python - <<'PY'
import site
import re
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
transformers_init = site_packages / "transformers" / "__init__.py"
transformers_utils_init = site_packages / "transformers" / "utils" / "__init__.py"
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

utils_marker = "# qwen35_sft_compat_utils"
if transformers_utils_init.exists():
    utils_text = transformers_utils_init.read_text()
    if utils_marker not in utils_text:
        utils_text += r'''

# qwen35_sft_compat_utils
def is_torch_sdpa_available():
    try:
        import torch
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")
    except Exception:
        return False

try:
    is_flash_attn_2_available
except NameError:
    def is_flash_attn_2_available():
        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False
'''
        transformers_utils_init.write_text(utils_text)
        print(f"patched {transformers_utils_init}")

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

attention = site_packages / "llamafactory" / "model" / "model_utils" / "attention.py"
old_attention_import = "from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available"
new_attention_import = """try:
    from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
except ImportError:
    import importlib.util
    def is_flash_attn_2_available():
        return importlib.util.find_spec("flash_attn") is not None
    def is_torch_sdpa_available():
        try:
            import torch
            return hasattr(torch.nn.functional, "scaled_dot_product_attention")
        except Exception:
            return False"""
if attention.exists():
    attention_text = attention.read_text()
    attention_text = attention_text.replace(
        "try:\nfrom transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available",
        new_attention_import,
    )
    attention_text = attention_text.replace(
        "try:\ntry:\n    from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available",
        new_attention_import,
    )
    attention_text, count = re.subn(
        r"(?m)^from transformers\.utils import is_flash_attn_2_available, is_torch_sdpa_available\s*$",
        new_attention_import,
        attention_text,
    )
    attention.write_text(attention_text)
    if count:
        print(f"patched {attention}")

    try:
        compile(attention.read_text(), str(attention), "exec")
    except IndentationError:
        broken = attention.read_text()
        broken = broken.replace(
            "try:\ntry:\n    from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available",
            new_attention_import,
        )
        broken = broken.replace(
            "try:\nfrom transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available",
            new_attention_import,
        )
        attention.write_text(broken)
        compile(attention.read_text(), str(attention), "exec")
        print(f"repaired indentation in {attention}")

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

    if not hasattr(_tf_utils, "is_torch_sdpa_available"):
        def is_torch_sdpa_available():
            try:
                import torch
                return hasattr(torch.nn.functional, "scaled_dot_product_attention")
            except Exception:
                return False
        _tf_utils.is_torch_sdpa_available = is_torch_sdpa_available

    if not hasattr(_tf_utils, "is_flash_attn_2_available"):
        def is_flash_attn_2_available():
            return importlib.util.find_spec("flash_attn") is not None
        _tf_utils.is_flash_attn_2_available = is_flash_attn_2_available

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
from transformers.utils import is_torch_sdpa_available
from trl import AutoModelForCausalLMWithValueHead
import llamafactory
from llamafactory.model.loader import AutoModelForVision2Seq
print("transformers", transformers.__version__)
print("qwen3_5 class", Qwen3_5ForCausalLM.__name__)
print("vision fallback", AutoModelForVision2Seq.__name__)
print("sdpa available", is_torch_sdpa_available())
print("imports ok")
PY
