#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--probe-output", required=True)
    parser.add_argument("--probe-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if len(df) <= args.probe_size:
        raise ValueError(f"Need more than probe_size rows. rows={len(df)} probe_size={args.probe_size}")

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    probe = shuffled.iloc[: args.probe_size].reset_index(drop=True)
    train = shuffled.iloc[args.probe_size :].reset_index(drop=True)

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.train_output, index=False)
    probe.to_parquet(args.probe_output, index=False)

    print(f"input rows: {len(df)}")
    print(f"train rows: {len(train)} -> {args.train_output}")
    print(f"probe rows: {len(probe)} -> {args.probe_output}")


if __name__ == "__main__":
    main()
