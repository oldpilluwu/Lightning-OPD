"""LlamaFactory-compatible Qwen3 SFT data loading for native TorchTitan."""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

IGNORE_INDEX = -100
LLAMAFACTORY_PACKING_BATCH_SIZE = 1000
QWEN3_IM_START = "<|im_start|>"
QWEN3_IM_END = "<|im_end|>"
QWEN3_EMPTY_THOUGHT = "<think>\n\n</think>\n\n"

EncodedExample = tuple[list[int], list[int], list[int]]
PackedExample = tuple[list[int], list[int], list[int]]


def _parquet_files(dataset_path: str) -> list[str]:
    path = Path(dataset_path).expanduser()
    if path.is_file() and path.suffix == ".parquet":
        return [str(path.resolve())]
    if path.is_dir():
        files = sorted(str(item.resolve()) for item in path.rglob("*.parquet"))
        if files:
            return files
    raise ValueError(
        "training.dataset_path must be a parquet file or a directory containing "
        f"parquet files; got {dataset_path!r}"
    )


def _normalize_messages(
    sample: dict[str, Any],
) -> tuple[str | None, list[dict[str, str]]]:
    """Apply LlamaFactory's ShareGPT SFT structural requirements."""
    raw_messages = sample.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Each SFT row must contain a non-empty 'messages' list")

    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}] has unsupported role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content must be a string")
        messages.append({"role": role, "content": content})

    system = None
    if messages[0]["role"] == "system":
        system = messages.pop(0)["content"]
    if any(message["role"] == "system" for message in messages):
        raise ValueError("A system message is only supported as the first message")
    if not messages or len(messages) % 2 != 0:
        raise ValueError("SFT messages must contain complete user/assistant pairs")

    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError(
                f"messages alternate user/assistant after the optional system message; "
                f"expected {expected_role!r} at position {index}, got {message['role']!r}"
            )
    return system, messages


def _infer_seqlen(
    source_len: int,
    target_len: int,
    cutoff_len: int,
) -> tuple[int, int]:
    """Match LlamaFactory's prompt/response-aware SFT truncation."""
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


def _greedy_knapsack_indices(
    lengths: list[int],
    capacity: int,
) -> list[list[int]]:
    """Return the same greedy length-based packs as LlamaFactory."""
    if any(length <= 0 or length > capacity for length in lengths):
        raise ValueError("Every encoded example must fit within the packing capacity")

    length_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, length in enumerate(lengths):
        length_to_indices[length].append(index)

    remaining_lengths = sorted(lengths)
    knapsacks: list[list[int]] = []
    while remaining_lengths:
        remaining_capacity = capacity
        knapsack: list[int] = []
        while True:
            fit_index = bisect.bisect(remaining_lengths, remaining_capacity) - 1
            if fit_index < 0:
                break
            length = remaining_lengths.pop(fit_index)
            remaining_capacity -= length
            knapsack.append(length_to_indices[length].pop())
        knapsacks.append(knapsack)
    return knapsacks


class PackedQwen3SFTDataset(IterableDataset, Stateful):
    """Reproduce LlamaFactory's Qwen3 SFT encoding and packing semantics.

    The configured ``seq_len`` is the user-visible LlamaFactory ``cutoff_len``.
    With packing enabled, LlamaFactory reserves one token, packs examples into
    ``seq_len - 1`` tokens, then pads to ``seq_len``. Position IDs restart at
    zero for every conversation. Labels are shifted here because TorchTitan's
    cross-entropy loss expects already-aligned targets.
    """

    def __init__(
        self,
        source: Dataset,
        tokenizer_path: str,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        seed: int,
        infinite: bool,
        packing_batch_size: int = LLAMAFACTORY_PACKING_BATCH_SIZE,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 for packed causal SFT")
        if packing_batch_size < 1:
            raise ValueError("packing_batch_size must be positive")

        self._source = source
        self._seq_len = seq_len
        self._packing_capacity = seq_len - 1
        self._packing_batch_size = packing_batch_size
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._seed = seed
        self._infinite = infinite

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=False,
            local_files_only=True,
        )
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError("Qwen3 tokenizer has neither a pad token nor an EOS token")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._pad_token_id = int(self._tokenizer.pad_token_id)
        for marker in (QWEN3_IM_START, QWEN3_IM_END):
            marker_ids = self._tokenizer.encode(marker, add_special_tokens=False)
            if len(marker_ids) != 1:
                raise ValueError(
                    f"Qwen3 tokenizer must encode {marker!r} as one special token"
                )

        self._epoch = 0
        self._sample_idx = 0
        self._batch_start_idx = 0
        self._batch_source_count = 0
        self._pack_idx = 0
        self._ready_packs: list[PackedExample] = []
        self._data = self._build_epoch_data()

    def _build_epoch_data(self) -> Dataset:
        shuffled = self._source.shuffle(seed=self._seed + self._epoch)
        data = split_dataset_by_node(
            shuffled,
            rank=self._dp_rank,
            world_size=self._dp_world_size,
        )
        if not isinstance(data, Dataset):
            raise TypeError("PackedQwen3SFTDataset requires a map-style Hugging Face dataset")
        return data

    def _encode_text(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self._tokenizer.encode(text, add_special_tokens=False)
        ]

    def _encode_example(self, sample: dict[str, Any]) -> EncodedExample:
        system, messages = _normalize_messages(sample)
        input_ids: list[int] = []
        labels: list[int] = []
        total_length = 0

        for turn_index in range(0, len(messages), 2):
            user = messages[turn_index]
            assistant = messages[turn_index + 1]

            source_text = ""
            if turn_index == 0 and system:
                source_text += (
                    f"{QWEN3_IM_START}system\n{system}{QWEN3_IM_END}\n"
                )
            source_text += (
                f"{QWEN3_IM_START}user\n{user['content']}{QWEN3_IM_END}\n"
                f"{QWEN3_IM_START}assistant\n"
            )

            assistant_content = assistant["content"]
            if "<think>" not in assistant_content and "</think>" not in assistant_content:
                assistant_content = QWEN3_EMPTY_THOUGHT + assistant_content
            target_text = f"{assistant_content}{QWEN3_IM_END}\n"

            source_ids = self._encode_text(source_text)
            target_ids = self._encode_text(target_text)
            remaining = self._packing_capacity - total_length
            if remaining <= 0:
                break
            source_len, target_len = _infer_seqlen(
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

        if not input_ids:
            raise ValueError("SFT example produced no tokens")
        if not any(label != IGNORE_INDEX for label in labels):
            raise ValueError("SFT example produced no assistant labels")
        positions = list(range(len(input_ids)))
        return input_ids, labels, positions

    def _encode_source_batch(
        self,
        start_idx: int,
        source_count: int,
    ) -> tuple[list[EncodedExample], int]:
        encoded: list[EncodedExample] = []
        consumed = 0
        for sample in self._data.skip(start_idx):
            if consumed >= source_count:
                break
            encoded.append(self._encode_example(sample))
            consumed += 1
        return encoded, consumed

    def _pack_examples(
        self,
        encoded_examples: list[EncodedExample],
    ) -> list[PackedExample]:
        if not encoded_examples:
            return []
        knapsacks = _greedy_knapsack_indices(
            [len(example[0]) for example in encoded_examples],
            self._packing_capacity,
        )
        packed_examples: list[PackedExample] = []
        for knapsack in knapsacks:
            packed_input_ids: list[int] = []
            packed_labels: list[int] = []
            packed_positions: list[int] = []
            for index in knapsack:
                input_ids, labels, positions = encoded_examples[index]
                packed_input_ids.extend(input_ids)
                packed_labels.extend(labels)
                packed_positions.extend(positions)

            pad_length = self._seq_len - len(packed_input_ids)
            if pad_length < 1:
                raise RuntimeError("LlamaFactory packing must reserve at least one pad token")
            packed_input_ids.extend([self._pad_token_id] * pad_length)
            packed_labels.extend([IGNORE_INDEX] * pad_length)
            packed_positions.extend([0] * pad_length)

            # Hugging Face causal-LM loss shifts logits and labels internally.
            # TorchTitan does not, so align each logit with the following label.
            shifted_labels = packed_labels[1:] + [IGNORE_INDEX]
            if not any(label != IGNORE_INDEX for label in shifted_labels):
                raise RuntimeError("Packed SFT block has no assistant labels")
            if not (
                len(packed_input_ids)
                == len(shifted_labels)
                == len(packed_positions)
                == self._seq_len
            ):
                raise RuntimeError("Packed SFT tensors do not match training.seq_len")
            packed_examples.append(
                (packed_input_ids, shifted_labels, packed_positions)
            )
        return packed_examples

    def _prepare_next_batch(self) -> bool:
        encoded, consumed = self._encode_source_batch(
            self._sample_idx,
            self._packing_batch_size,
        )
        if consumed == 0:
            return False

        self._batch_start_idx = self._sample_idx
        self._batch_source_count = consumed
        self._sample_idx += consumed
        self._ready_packs = self._pack_examples(encoded)
        self._pack_idx = 0
        return True

    def _rebuild_current_batch(self) -> None:
        if self._batch_source_count == 0:
            self._ready_packs = []
            return
        encoded, consumed = self._encode_source_batch(
            self._batch_start_idx,
            self._batch_source_count,
        )
        if consumed != self._batch_source_count:
            raise RuntimeError(
                "Could not reconstruct the saved LlamaFactory packing batch"
            )
        self._ready_packs = self._pack_examples(encoded)
        if self._pack_idx > len(self._ready_packs):
            raise ValueError("Saved pack index exceeds reconstructed packing batch")

    @staticmethod
    def _to_tensors(
        packed_example: PackedExample,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        input_ids, labels, positions = packed_example
        return {
            "input": torch.tensor(input_ids, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
        }, torch.tensor(labels, dtype=torch.long)

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            if self._pack_idx < len(self._ready_packs):
                packed_example = self._ready_packs[self._pack_idx]
                self._pack_idx += 1
                yield self._to_tensors(packed_example)
                continue

            self._ready_packs = []
            self._batch_source_count = 0
            self._pack_idx = 0
            if self._prepare_next_batch():
                continue

            if not self._infinite:
                logger.warning("Qwen3 SFT dataset has run out of data")
                return

            self._epoch += 1
            self._sample_idx = 0
            self._batch_start_idx = 0
            self._data = self._build_epoch_data()
            logger.warning("Qwen3 SFT dataset is starting epoch %d", self._epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "epoch": self._epoch,
            "sample_idx": self._sample_idx,
            "batch_start_idx": self._batch_start_idx,
            "batch_source_count": self._batch_source_count,
            "pack_idx": self._pack_idx,
            "dp_rank": self._dp_rank,
            "dp_world_size": self._dp_world_size,
            "packing_batch_size": self._packing_batch_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict.get("format_version") != 2:
            raise ValueError(
                "This dataloader cannot resume checkpoints made by the older "
                "non-LlamaFactory-compatible packing implementation"
            )
        if state_dict["dp_rank"] != self._dp_rank:
            raise ValueError("Cannot load SFT dataloader state for a different DP rank")
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError("SFT dataloader resharding is not supported")
        if state_dict["packing_batch_size"] != self._packing_batch_size:
            raise ValueError("Cannot change packing_batch_size when resuming")

        self._epoch = int(state_dict["epoch"])
        self._sample_idx = int(state_dict["sample_idx"])
        self._batch_start_idx = int(state_dict["batch_start_idx"])
        self._batch_source_count = int(state_dict["batch_source_count"])
        self._pack_idx = int(state_dict["pack_idx"])
        self._data = self._build_epoch_data()
        self._rebuild_current_batch()


def build_sft_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build the Lightning-OPD parquet dataloader for TorchTitan."""
    del tokenizer  # LlamaFactory-compatible encoding uses the full HF tokenizer.

    dataset_path = job_config.training.dataset_path
    if not dataset_path:
        raise ValueError("training.dataset_path is required for qwen3_sft")
    parquet_files = _parquet_files(dataset_path)
    source = load_dataset(
        "parquet",
        data_files={"train": parquet_files},
        split="train",
    )
    if not isinstance(source, Dataset):
        raise TypeError("Expected a map-style parquet dataset")
    missing_columns = {"messages"} - set(source.column_names)
    if missing_columns:
        raise ValueError(
            f"SFT parquet is missing required columns: {sorted(missing_columns)}"
        )

    dataloader_config = job_config.training.dataloader
    if dataloader_config.num_workers != 0:
        raise ValueError(
            "The checkpointable packed SFT loader requires num_workers=0; "
            "multiple iterable workers would duplicate samples"
        )

    seed = job_config.debug.seed if job_config.debug.seed is not None else 42
    dataset = PackedQwen3SFTDataset(
        source=source,
        tokenizer_path=job_config.model.hf_assets_path,
        seq_len=job_config.training.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        seed=seed,
        infinite=infinite,
    )
    dataloader_kwargs = {
        **asdict(dataloader_config),
        "batch_size": job_config.training.local_batch_size,
    }
    return ParallelAwareDataloader(
        dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
