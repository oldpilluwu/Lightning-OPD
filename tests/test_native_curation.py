import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trainium.sft_data_generation_native.pipeline import (
    generate_static,
    load_dataset,
    render_prompt,
    top_p_sample_cpu,
)


class FakeTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        self.last_messages = messages
        return f"rendered:{messages[-1]['content']}"


class FakeStaticCache:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.calls = []

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls.append((tuple(input_ids.shape), tuple(attention_mask.shape)))
        token = 2 if len(self.calls) == 1 else 3
        logits = torch.zeros(*input_ids.shape, 5)
        logits[..., token] = 10
        return SimpleNamespace(logits=logits)


class RecordingCaller:
    def __init__(self, model, calls):
        self.model = model
        self.calls = calls

    def __call__(self, **kwargs):
        self.calls.append(tuple(kwargs["input_ids"].shape))
        return self.model(**kwargs)


class NativeCurationTests(unittest.TestCase):
    def test_greedy_sampling_uses_fp32_cpu(self):
        logits = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.bfloat16)
        sampled = top_p_sample_cpu(logits, temperature=0.0, top_p=1.0)
        self.assertEqual(sampled.device.type, "cpu")
        self.assertEqual(sampled.tolist(), [1])

    def test_render_prompt_preserves_chat_messages(self):
        tokenizer = FakeTokenizer()
        prompt = [{"role": "system", "content": "reason"}, {"role": "user", "content": "2+2?"}]
        self.assertEqual(render_prompt(tokenizer, prompt), "rendered:2+2?")
        self.assertEqual(tokenizer.last_messages, prompt)

    def test_jsonl_loader_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"prompt": "hello"}) + "\n\n", encoding="utf-8")
            self.assertEqual(load_dataset(str(path)), [{"prompt": "hello"}])

    def test_static_decode_keeps_decode_shapes_fixed(self):
        model = FakeModel()
        with patch("transformers.StaticCache", FakeStaticCache):
            generated = generate_static(
                model,
                torch.tensor([[0, 0, 7, 8], [0, 4, 5, 6]]),
                torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
                max_new_tokens=2,
                temperature=0.0,
                top_p=1.0,
                eos_token_ids={3},
                pad_token_id=0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        self.assertEqual(generated, [[2, 3], [2, 3]])
        self.assertEqual(model.calls, [((2, 4), (2, 4)), ((2, 1), (2, 6))])

    def test_static_decode_uses_separate_prefill_and_decode_callers(self):
        model = FakeModel()
        prefill_calls = []
        decode_calls = []
        object.__setattr__(model, "_native_prefill_model", RecordingCaller(model, prefill_calls))
        object.__setattr__(model, "_native_decode_model", RecordingCaller(model, decode_calls))
        with patch("transformers.StaticCache", FakeStaticCache):
            generate_static(
                model,
                torch.tensor([[7, 8]]),
                torch.tensor([[1, 1]]),
                max_new_tokens=2,
                temperature=0.0,
                top_p=1.0,
                eos_token_ids={3},
                pad_token_id=0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        self.assertEqual(prefill_calls, [(1, 2)])
        self.assertEqual(decode_calls, [(1, 1)])


if __name__ == "__main__":
    unittest.main()
