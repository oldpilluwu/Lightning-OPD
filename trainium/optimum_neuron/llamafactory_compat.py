"""Exact Qwen3 SFT preprocessing used by the Lightning-OPD LLaMA-Factory run.

The implementation mirrors the relevant LLaMA-Factory code paths:

* ``template: qwen3`` with reasoning enabled
* ``train_on_prompt: false`` and ``mask_history: false``
* prompt-aware source/target truncation
* ``packing: true`` and ``neat_packing: false``
* the one-token packing reservation used by ``DataArguments``

Reference LLaMA-Factory commit:
9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

IGNORE_INDEX = -100
LLAMAFACTORY_REFERENCE_COMMIT = "9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d"
LLAMAFACTORY_PREPROCESSING_BATCH_SIZE = 1000
QWEN3_IM_START = "<|im_start|>"
QWEN3_IM_END = "<|im_end|>"
QWEN3_THINK_OPEN = "<think>\n"
QWEN3_THINK_CLOSE = "\n</think>\n\n"


def fix_qwen3_tokenizer(tokenizer: Any) -> None:
    """Apply LLaMA-Factory's qwen3 special-token fix."""
    if tokenizer.eos_token != QWEN3_IM_END:
        tokenizer.add_special_tokens({"eos_token": QWEN3_IM_END})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def infer_seqlen(
    source_len: int,
    target_len: int,
    cutoff_len: int,
) -> tuple[int, int]:
    """Match LLaMA-Factory's prompt/response-aware truncation."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    """Match LLaMA-Factory's greedy packing algorithm."""
    numbers.sort()
    knapsacks: list[list[int]] = []
    while numbers:
        current_knapsack: list[int] = []
        remaining_capacity = capacity
        while True:
            index = bisect.bisect(numbers, remaining_capacity) - 1
            if index < 0:
                break
            remaining_capacity -= numbers[index]
            current_knapsack.append(numbers.pop(index))
        knapsacks.append(current_knapsack)
    return knapsacks


def _messages_as_sequence(raw_messages: Any) -> list[Mapping[str, Any]]:
    if isinstance(raw_messages, Mapping):
        role_values = raw_messages.get("role", raw_messages.get("from"))
        content_values = raw_messages.get("content", raw_messages.get("value"))
        if isinstance(role_values, Sequence) and isinstance(content_values, Sequence):
            return [
                {"role": role, "content": content}
                for role, content in zip(role_values, content_values, strict=True)
            ]
    if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
        return list(raw_messages)
    raise ValueError("messages must be a list of message objects")


def normalize_messages(raw_messages: Any) -> tuple[str, list[dict[str, str]]]:
    """Align OpenAI or ShareGPT messages like LLaMA-Factory's converter."""
    role_aliases = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
    }
    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(_messages_as_sequence(raw_messages)):
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        raw_role = raw_message.get("role", raw_message.get("from"))
        content = raw_message.get("content", raw_message.get("value"))
        role = role_aliases.get(raw_role)
        if role is None:
            raise ValueError(f"messages[{index}] has unsupported role {raw_role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}] content must be a string")
        messages.append({"role": role, "content": content})

    system = ""
    if messages and messages[0]["role"] == "system":
        system = messages.pop(0)["content"]

    if not messages or len(messages) % 2 != 0:
        raise ValueError("messages must contain complete user/assistant pairs")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(
                f"expected role {expected!r} at aligned position {index}, "
                f"found {message['role']!r}"
            )
    return system, messages


@dataclass
class Qwen3LlamaFactoryPackedProcessor:
    """Callable compatible with ``datasets.Dataset.map(batched=True)``."""

    tokenizer: Any
    cutoff_len: int = 16384

    def __post_init__(self) -> None:
        if self.cutoff_len < 2:
            raise ValueError("cutoff_len must be at least 2")
        fix_qwen3_tokenizer(self.tokenizer)
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer has no pad token after Qwen3 token repair")
        # LLaMA-Factory's DataArguments subtracts one when packing=True.
        self.packing_capacity = self.cutoff_len - 1

    def _encode_text(self, text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.tokenizer.encode(text, add_special_tokens=False)
        ]

    def encode_example(self, raw_messages: Any) -> tuple[list[int], list[int]]:
        """Produce the exact pre-packing ``input_ids`` and ``labels``."""
        system, messages = normalize_messages(raw_messages)
        input_ids: list[int] = []
        labels: list[int] = []
        total_length = 0

        for turn_index in range(0, len(messages), 2):
            user = messages[turn_index]
            assistant = messages[turn_index + 1]

            # LLaMA-Factory tokenizes formatter slots separately. Keeping system
            # and user as separate calls preserves tokenizer boundary behavior.
            source_ids: list[int] = []
            if turn_index == 0 and system:
                source_ids.extend(
                    self._encode_text(
                        f"{QWEN3_IM_START}system\n{system}{QWEN3_IM_END}\n"
                    )
                )
            source_ids.extend(
                self._encode_text(
                    f"{QWEN3_IM_START}user\n{user['content']}{QWEN3_IM_END}\n"
                    f"{QWEN3_IM_START}assistant\n"
                )
            )

            target_ids = self._encode_text(
                f"{assistant['content']}{QWEN3_IM_END}\n"
            )
            if (
                QWEN3_THINK_OPEN.strip() not in assistant["content"]
                and QWEN3_THINK_CLOSE.strip() not in assistant["content"]
            ):
                empty_thought_ids = self._encode_text(
                    QWEN3_THINK_OPEN + QWEN3_THINK_CLOSE
                )
                target_ids = empty_thought_ids + target_ids

            remaining = self.packing_capacity - total_length
            if remaining <= 0:
                break
            source_len, target_len = infer_seqlen(
                len(source_ids),
                len(target_ids),
                remaining,
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            input_ids.extend(source_ids)
            input_ids.extend(target_ids)
            labels.extend([IGNORE_INDEX] * source_len)
            labels.extend(target_ids)
            total_length += source_len + target_len

        return input_ids, labels

    def __call__(self, examples: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        """Encode and pack one LLaMA-Factory preprocessing batch."""
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        lengths: list[int] = []
        length_to_indexes: dict[int, list[int]] = defaultdict(list)

        for raw_messages in examples["messages"]:
            try:
                input_ids, labels = self.encode_example(raw_messages)
            except (KeyError, TypeError, ValueError):
                # LLaMA-Factory's converter represents malformed examples as
                # empty aligned rows, which its SFT processor then drops.
                continue
            if not input_ids or len(input_ids) > self.packing_capacity:
                continue
            index = len(batch_input_ids)
            length = len(input_ids)
            lengths.append(length)
            length_to_indexes[length].append(index)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        model_inputs: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [],
            "labels": [],
        }
        for knapsack in greedy_knapsack(lengths, self.packing_capacity):
            packed_input_ids: list[int] = []
            packed_labels: list[int] = []
            packed_position_ids: list[int] = []
            for length in knapsack:
                index = length_to_indexes[length].pop()
                sample_input_ids = batch_input_ids[index]
                packed_input_ids.extend(sample_input_ids)
                packed_labels.extend(batch_labels[index])
                packed_position_ids.extend(range(len(sample_input_ids)))

            pad_length = self.cutoff_len - len(packed_input_ids)
            if pad_length < 1:
                raise RuntimeError("Packed Qwen3 examples must reserve one pad token")
            packed_input_ids.extend([int(self.tokenizer.pad_token_id)] * pad_length)
            packed_labels.extend([IGNORE_INDEX] * pad_length)
            packed_position_ids.extend([0] * pad_length)

            if not (
                len(packed_input_ids)
                == len(packed_labels)
                == len(packed_position_ids)
                == self.cutoff_len
            ):
                raise RuntimeError("Packed tensors do not match cutoff_len")

            model_inputs["input_ids"].append(packed_input_ids)
            # With neat_packing=False, LLaMA-Factory deliberately uses ones
            # for both real tokens and right-padding.
            model_inputs["attention_mask"].append([1] * self.cutoff_len)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)

        return model_inputs
