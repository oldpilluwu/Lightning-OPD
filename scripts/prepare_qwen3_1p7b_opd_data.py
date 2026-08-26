#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepare deterministic OPD train and post-training evaluation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

DATASETS = {
    # Prompt-only 17,398-row mirror used by this repository. The upstream
    # BytedTsinghua-SIA repo currently exposes ~1.79M trajectory rows.
    "dapo": ("zhuzilin/dapo-math-17k", "train"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "aime24": ("math-ai/aime24", "test"),
    "aime25": ("math-ai/aime25", "test"),
    "amc23": ("math-ai/amc23", "test"),
}
OVERLAP_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _last_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text or "")
    return matches[-1] if matches else str(text)


def _prompt_messages(value) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in value]
    return [{"role": "user", "content": str(value)}]


def _normalize_dapo(row: dict) -> dict:
    reward_model = row.get("reward_model") or {}
    extra_info = row.get("extra_info") or {}
    prompt = _prompt_messages(row["prompt"])
    stable_id = extra_info.get("index", row.get("id"))
    if stable_id is None:
        payload = json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode("utf-8")
        stable_id = hashlib.sha256(payload).hexdigest()[:24]
    return {
        "id": str(stable_id),
        "prompt": prompt,
        "label": str(reward_model.get("ground_truth", row.get("label", ""))),
        "source": "dapo",
    }


def _normalize_eval(name: str, row: dict, index: int) -> dict:
    prompt = row.get("problem", row.get("question"))
    answer = row.get("answer")
    if answer is None:
        answer = _last_boxed(row.get("solution", ""))
    return {
        "id": str(row.get("unique_id", row.get("id", f"{name}-{index}"))),
        "prompt": str(prompt),
        "label": str(answer),
        "source": name,
        "subject": row.get("subject"),
        "level": row.get("level"),
    }


def _question_from_messages(messages: list[dict[str, str]]) -> str:
    content = "\n".join(message["content"] for message in messages if message["role"] == "user")
    sections = [section.strip() for section in content.split("\n\n") if section.strip()]
    candidates = [
        section
        for section in sections
        if not section.lower().startswith(("solve the following", "remember to", "the last line"))
    ]
    return max(candidates or sections or [content], key=len)


@torch.inference_mode()
def _embed_texts(model, tokenizer, texts: list[str], device: str, batch_size: int = 64) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(embeddings)


def add_training_overlap_flags(
    api: HfApi,
    train_rows: list[dict],
    eval_rows: dict[str, list[dict]],
    *,
    requested_revision: str,
    device: str,
    threshold: float,
) -> str:
    """Flag exact and MPNet-cosine overlap without removing official benchmark rows."""
    model_revision = api.model_info(OVERLAP_MODEL, revision=requested_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(OVERLAP_MODEL, revision=model_revision)
    model = AutoModel.from_pretrained(OVERLAP_MODEL, revision=model_revision).to(device).eval()
    train_texts = [_question_from_messages(row["prompt"]) for row in train_rows]
    train_embeddings = _embed_texts(model, tokenizer, train_texts, device)
    exact_lookup = {re.sub(r"\s+", " ", text).strip().casefold(): index for index, text in enumerate(train_texts)}

    for rows in eval_rows.values():
        embeddings = _embed_texts(model, tokenizer, [row["prompt"] for row in rows], device)
        for start in range(0, len(rows), 128):
            similarities = embeddings[start : start + 128] @ train_embeddings.T
            scores, indices = similarities.max(dim=-1)
            for offset, (score, train_index) in enumerate(zip(scores.tolist(), indices.tolist(), strict=True)):
                row = rows[start + offset]
                normalized = re.sub(r"\s+", " ", row["prompt"]).strip().casefold()
                row["training_overlap_exact"] = normalized in exact_lookup
                row["training_overlap_cosine"] = float(score)
                row["training_overlap_semantic"] = score >= threshold
                row["nearest_training_prompt_id"] = train_rows[train_index]["id"]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_revision


def stratified_math500(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin a seeded shuffle of subject/level strata."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("subject")), str(row.get("level")))].append(row)
    queues = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        queues.append(deque(groups[key]))
    selected = []
    while queues and len(selected) < count:
        next_queues = []
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    if len(selected) != count:
        raise ValueError(f"Requested {count} MATH-500 diagnostics but selected {len(selected)}")
    return selected


def _resolve_and_load(api: HfApi, repo_id: str, split: str, revision: str) -> tuple[Dataset, str]:
    resolved_revision = api.dataset_info(repo_id, revision=revision).sha
    dataset = load_dataset(repo_id, split=split, revision=resolved_revision)
    return dataset, resolved_revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets/qwen3-1.7b-opd",
    )
    parser.add_argument("--revision", default="main", help="Hub revision resolved to immutable commit SHAs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dapo-diagnostic-count", type=int, default=64)
    parser.add_argument("--math-diagnostic-count", type=int, default=64)
    parser.add_argument("--semantic-overlap-threshold", type=float, default=0.6)
    parser.add_argument("--overlap-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-semantic-overlap", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    loaded = {}
    revisions = {}
    fingerprints = {}
    for name, (repo_id, split) in DATASETS.items():
        dataset, resolved_revision = _resolve_and_load(api, repo_id, split, args.revision)
        loaded[name] = dataset
        revisions[name] = {"repo_id": repo_id, "revision": resolved_revision, "split": split}
        fingerprints[name] = dataset._fingerprint

    dapo_rows = [_normalize_dapo(row) for row in loaded["dapo"]]
    rng = random.Random(args.seed)
    shuffled_indices = list(range(len(dapo_rows)))
    rng.shuffle(shuffled_indices)
    held_out = set(shuffled_indices[: args.dapo_diagnostic_count])
    dapo_diagnostic = [dapo_rows[index] for index in shuffled_indices[: args.dapo_diagnostic_count]]
    dapo_train = [row for index, row in enumerate(dapo_rows) if index not in held_out]

    eval_rows = {
        name: [_normalize_eval(name, row, index) for index, row in enumerate(loaded[name])]
        for name in ("math500", "aime24", "aime25", "amc23")
    }
    overlap_model_revision = None
    if not args.skip_semantic_overlap:
        overlap_model_revision = add_training_overlap_flags(
            api,
            dapo_train,
            eval_rows,
            requested_revision=args.revision,
            device=args.overlap_device,
            threshold=args.semantic_overlap_threshold,
        )
    math_diagnostic = stratified_math500(eval_rows["math500"], args.math_diagnostic_count, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "dapo_train.jsonl", dapo_train)
    _write_jsonl(args.output_dir / "dapo_diagnostic.jsonl", dapo_diagnostic)
    _write_jsonl(args.output_dir / "math500_diagnostic.jsonl", math_diagnostic)
    for name, rows in eval_rows.items():
        _write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "datasets": revisions,
        "fingerprints": fingerprints,
        "overlap_analysis": {
            "model": OVERLAP_MODEL if overlap_model_revision else None,
            "revision": overlap_model_revision,
            "cosine_threshold": args.semantic_overlap_threshold,
        },
        "counts": {
            "dapo_original": len(dapo_rows),
            "dapo_train": len(dapo_train),
            "dapo_diagnostic": len(dapo_diagnostic),
            "math500_diagnostic": len(math_diagnostic),
            **{name: len(rows) for name, rows in eval_rows.items()},
        },
        "dapo_diagnostic_ids": [row["id"] for row in dapo_diagnostic],
        "math500_diagnostic_ids": [row["id"] for row in math_diagnostic],
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
