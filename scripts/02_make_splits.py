"""
02_make_splits.py
Split events.jsonl into train / val / test sets.

Two strategies supported:
  --strategy random   : stratified random split (default 70/15/15)
  --strategy loto     : leave-one-topic-out (each run: one topic = test)

Output: data/processed/{train,val,test}.jsonl
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


def load_events(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(events: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({len(events)} events)")


def random_split(events: list[dict], train_ratio=0.7, val_ratio=0.15, seed=42):
    """Stratified random split by label."""
    random.seed(seed)
    by_label = defaultdict(list)
    for e in events:
        by_label[e["label"]].append(e)

    train, val, test = [], [], []
    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test


def loto_split(events: list[dict], test_topic: str):
    """Leave-one-topic-out: one topic is test, rest split into train/val."""
    train_val = [e for e in events if e["topic"] != test_topic]
    test = [e for e in events if e["topic"] == test_topic]

    random.seed(42)
    random.shuffle(train_val)
    n = len(train_val)
    n_val = int(n * 0.15)
    val = train_val[:n_val]
    train = train_val[n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/events.jsonl")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--strategy", choices=["random", "loto"], default="random")
    parser.add_argument("--test_topic", default=None,
                        help="Topic name for leave-one-topic-out (required if strategy=loto)")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    events = load_events(Path(args.input))
    topics = sorted({e["topic"] for e in events})
    label_counts = defaultdict(int)
    for e in events:
        label_counts[e["label"]] += 1

    print(f"Loaded {len(events)} events")
    print(f"  rumour: {label_counts[1]}, non-rumour: {label_counts[0]}")
    print(f"  topics: {topics}")

    if args.strategy == "random":
        train, val, test = random_split(events, args.train_ratio, args.val_ratio, args.seed)
    else:
        if args.test_topic is None:
            print("ERROR: --test_topic required for loto strategy")
            return
        if args.test_topic not in topics:
            print(f"ERROR: test_topic '{args.test_topic}' not found. Available: {topics}")
            return
        train, val, test = loto_split(events, args.test_topic)

    out = Path(args.output_dir)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_jsonl(test, out / "test.jsonl")

    print(f"\nSplit complete (strategy={args.strategy})")
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
