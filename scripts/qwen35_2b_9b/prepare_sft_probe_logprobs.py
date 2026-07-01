#!/usr/bin/env python3

import argparse
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:13141/generate")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-response-len", type=int, default=2048)
    return parser.parse_args()


def build_rows(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    df = pd.read_parquet(args.input_parquet)
    rows = []

    for row in tqdm(df.itertuples(), total=len(df), desc="tokenize probe"):
        messages = row.messages
        user_messages = [m for m in messages if m["role"] != "assistant"]
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if assistant_msg is None:
            continue

        prompt = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(assistant_msg, add_special_tokens=False)
        if len(response_ids) > args.max_response_len:
            response_ids = response_ids[: args.max_response_len]
            assistant_msg = tokenizer.decode(response_ids, skip_special_tokens=False)

        rows.append(
            {
                "prompt": prompt,
                "prompt_tokens": prompt_ids,
                "response": assistant_msg,
                "response_tokens": response_ids,
                "teacher_log_probs": None,
            }
        )

    return rows


async def fetch_logprobs(session, teacher_url, full_ids, response_len):
    payload = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with session.post(teacher_url, json=payload) as resp:
        resp.raise_for_status()
        ret = await resp.json()
    all_lps = ret["meta_info"]["input_token_logprobs"]
    response_lps = [float(item[0]) for item in all_lps[1:]][-response_len:]
    if len(response_lps) != response_len:
        raise RuntimeError(f"Expected {response_len} logprobs, got {len(response_lps)}")
    return response_lps


async def process_all(args, rows):
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(rows), desc="teacher probe logprobs") as pbar:
            async def one(row):
                async with semaphore:
                    full_ids = [int(x) for x in row["prompt_tokens"]] + [int(x) for x in row["response_tokens"]]
                    row["teacher_log_probs"] = await fetch_logprobs(
                        session,
                        args.teacher_url,
                        full_ids,
                        len(row["response_tokens"]),
                    )
                    pbar.update(1)

            await asyncio.gather(*(one(row) for row in rows))


def main():
    args = parse_args()
    rows = build_rows(args)
    asyncio.run(process_all(args, rows))

    output = Path(args.output_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"wrote {len(rows)} probe rows -> {output}")


if __name__ == "__main__":
    main()
