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
modeling_auto = site_packages / "transformers" / "models" / "auto" / "modeling_auto.py"
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

modeling_marker = "# qwen35_sft_compat_vision_mapping"
if modeling_auto.exists():
    modeling_text = modeling_auto.read_text()
    if modeling_marker not in modeling_text:
        modeling_text += r'''

# qwen35_sft_compat_vision_mapping
try:
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
except NameError:
    try:
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    except NameError:
        MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
'''
        modeling_auto.write_text(modeling_text)
        print(f"patched {modeling_auto}")

utils_marker = "# qwen35_sft_compat_utils"
if transformers_utils_init.exists():
    utils_text = transformers_utils_init.read_text()
    if utils_marker not in utils_text:
        utils_text += r'''

# qwen35_sft_compat_utils
import importlib.util as _qwen35_importlib_util

def _qwen35_has_package(name):
    return _qwen35_importlib_util.find_spec(name) is not None

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

try:
    is_safetensors_available
except NameError:
    def is_safetensors_available():
        return _qwen35_has_package("safetensors")

try:
    is_tensorboard_available
except NameError:
    def is_tensorboard_available():
        return _qwen35_has_package("tensorboard") or _qwen35_has_package("tensorboardX")

try:
    is_wandb_available
except NameError:
    def is_wandb_available():
        return _qwen35_has_package("wandb")

try:
    is_datasets_available
except NameError:
    def is_datasets_available():
        return _qwen35_has_package("datasets")

try:
    is_peft_available
except NameError:
    def is_peft_available():
        return _qwen35_has_package("peft")
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
    target = "from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available"
    cleanup = []
    lines = attention_text.splitlines()
    i = 0
    inserted = False
    while i < len(lines):
        stripped = lines[i].strip()
        lookahead = "\n".join(lines[i : min(len(lines), i + 16)])
        if (
            stripped == "try:"
            and (
                target in lookahead
                or "def is_flash_attn_2_available():" in lookahead
                or "def is_torch_sdpa_available():" in lookahead
            )
        ):
            while i < len(lines):
                current = lines[i].strip()
                if current.startswith(("def configure_", "def print_", "logger =", "ATTN")):
                    break
                if current.startswith("from ") and target not in current and "flash_attn" not in current:
                    break
                i += 1
            if not inserted:
                cleanup.extend(new_attention_import.splitlines())
                inserted = True
            continue
        if stripped == target:
            if cleanup and cleanup[-1].strip() == "try:":
                cleanup.pop()
            if not inserted:
                cleanup.extend(new_attention_import.splitlines())
                inserted = True
            i += 1
            continue
        cleanup.append(lines[i])
        i += 1

    attention_text = "\n".join(cleanup) + "\n"
    count = 1 if inserted else 0
    if not inserted:
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

    if not hasattr(_tf_utils, "is_safetensors_available"):
        def is_safetensors_available():
            return importlib.util.find_spec("safetensors") is not None
        _tf_utils.is_safetensors_available = is_safetensors_available

    if not hasattr(_tf_utils, "is_tensorboard_available"):
        def is_tensorboard_available():
            return (
                importlib.util.find_spec("tensorboard") is not None
                or importlib.util.find_spec("tensorboardX") is not None
            )
        _tf_utils.is_tensorboard_available = is_tensorboard_available

    if not hasattr(_tf_utils, "is_wandb_available"):
        def is_wandb_available():
            return importlib.util.find_spec("wandb") is not None
        _tf_utils.is_wandb_available = is_wandb_available

    if not hasattr(_tf_utils, "is_datasets_available"):
        def is_datasets_available():
            return importlib.util.find_spec("datasets") is not None
        _tf_utils.is_datasets_available = is_datasets_available

    if not hasattr(_tf_utils, "is_peft_available"):
        def is_peft_available():
            return importlib.util.find_spec("peft") is not None
        _tf_utils.is_peft_available = is_peft_available

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
from transformers.utils import is_safetensors_available, is_torch_sdpa_available
from trl import AutoModelForCausalLMWithValueHead
from trl import DPOTrainer
import llamafactory
from llamafactory.model.loader import AutoModelForVision2Seq
print("transformers", transformers.__version__)
print("qwen3_5 class", Qwen3_5ForCausalLM.__name__)
print("vision fallback", AutoModelForVision2Seq.__name__)
print("sdpa available", is_torch_sdpa_available())
print("safetensors available", is_safetensors_available())
print("trl dpo", DPOTrainer.__name__)
print("imports ok")
PY
