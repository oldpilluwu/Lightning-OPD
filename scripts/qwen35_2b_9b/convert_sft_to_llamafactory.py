#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
}


def convert_messages(messages):
    converted = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            raise ValueError(f"Expected OpenAI-style message with role/content, got: {message}")
        converted.append({"from": ROLE_MAP.get(role, role), "value": content})
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "messages" not in df.columns:
        raise ValueError(f"Input parquet has no messages column: {args.input}")

    out = pd.DataFrame({"messages": [convert_messages(messages) for messages in df["messages"]]})
    if "tokens" in df.columns:
        out["tokens"] = df["tokens"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    print(f"wrote {len(out)} LLaMA-Factory rows -> {output}")


if __name__ == "__main__":
    main()
