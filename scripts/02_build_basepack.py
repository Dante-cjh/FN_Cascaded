"""
02_build_basepack.py
Build BasePack text representation for each event in the data splits.

BasePack is the unified input format for all three experiments.
Reply selection strategy (heuristic):
  1. Earliest 2-3 replies (chronological order)
  2. Longest 2-3 replies (by character count)
  3. One reply from each of up to 2 distinct propagation branches

Output: data/processed/basepack_{split}.jsonl
Each line:
{
  "event_id":        str,
  "label":           int,        # 0=True, 1=Fake
  "basepack_text":   str,        # formatted text for small model input
  "source_text":     str,
  "selected_replies": [str],
  "stats": {"num_replies": int, "max_depth": int, "num_branches": int, "time_span": str}
}
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# --------------------------------------------------------------------------- #
#  Branch map builder (reused from 05_pack_evidence.py)
# --------------------------------------------------------------------------- #

def build_branch_map(structure: dict) -> dict[str, str]:
    """Map tweet_id -> branch_root (first child of source tweet)."""
    branch_map: dict[str, str] = {}

    def _walk(node: dict, branch_root: str | None):
        if not isinstance(node, dict):
            return
        for tid, children in node.items():
            root = branch_root if branch_root else tid
            branch_map[tid] = root
            _walk(children, root)

    _walk(structure, None)
    return branch_map


# --------------------------------------------------------------------------- #
#  Reply selector
# --------------------------------------------------------------------------- #

def select_replies(replies: list[dict], structure: dict, max_replies: int = 8) -> list[str]:
    if not replies:
        return []

    timed = sorted(replies, key=lambda r: r.get("time", ""))

    selected_ids: set[str] = set()
    selected_texts: list[str] = []

    def add(r: dict):
        tid = r["tweet_id"]
        text = r.get("text", "").strip()
        if tid not in selected_ids and text:
            selected_ids.add(tid)
            selected_texts.append(text)

    # 1. Earliest 3 replies
    for r in timed[:3]:
        add(r)

    # 2. Longest 3 (excluding already selected)
    by_len = sorted(replies, key=lambda r: len(r.get("text", "")), reverse=True)
    count = 0
    for r in by_len:
        if r["tweet_id"] not in selected_ids:
            add(r)
            count += 1
        if count >= 3:
            break

    # 3. One per branch (up to 2 branches)
    branch_map = build_branch_map(structure)
    seen_branches: set[str] = set()
    for r in timed:
        if len(seen_branches) >= 2:
            break
        br = branch_map.get(r["tweet_id"], r["tweet_id"])
        if br not in seen_branches and r["tweet_id"] not in selected_ids:
            seen_branches.add(br)
            add(r)

    return selected_texts[:max_replies]


# --------------------------------------------------------------------------- #
#  BasePack text formatter
# --------------------------------------------------------------------------- #

def format_basepack(source_text: str, selected_replies: list[str], stats: dict) -> str:
    lines = []
    lines.append("[SOURCE]")
    lines.append(source_text)
    lines.append("")

    for i, reply in enumerate(selected_replies, 1):
        lines.append(f"[REPLY_{i}]")
        lines.append(reply)
        lines.append("")

    lines.append("[STATS]")
    lines.append(f"reply_count={stats.get('num_replies', 0)}")
    lines.append(f"max_depth={stats.get('max_depth', 0)}")
    lines.append(f"num_branches={stats.get('num_branches', 0)}")
    lines.append(f"time_span={stats.get('time_span', 'unknown')}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def build_basepack_for_split(input_path: Path, output_path: Path, max_replies: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            e = json.loads(line)

            selected = select_replies(
                e.get("replies", []),
                e.get("structure", {}),
                max_replies,
            )
            stats = e.get("meta", {})
            basepack_text = format_basepack(e["source_text"], selected, stats)

            record = {
                "event_id": e["event_id"],
                "label": e["label"],
                "basepack_text": basepack_text,
                "source_text": e["source_text"],
                "selected_replies": selected,
                "stats": {
                    "num_replies": stats.get("num_replies", 0),
                    "max_depth": stats.get("max_depth", 0),
                    "num_branches": stats.get("num_branches", 0),
                    "time_span": stats.get("time_span", "unknown"),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    count = sum(1 for _ in open(output_path))
    print(f"  -> {output_path}  ({count} events)")


def main():
    parser = argparse.ArgumentParser(description="Build BasePack text representations")
    parser.add_argument("--input_dir", default="data/processed",
                        help="Directory containing train/val/test.jsonl splits")
    parser.add_argument("--output_dir", default="data/processed",
                        help="Directory for basepack_{split}.jsonl outputs")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max_replies", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"basepack_{split}.jsonl"
        if not input_path.exists():
            print(f"[WARN] Not found: {input_path}, skipping.")
            continue
        print(f"Building BasePack for {split}...")
        build_basepack_for_split(input_path, output_path, args.max_replies)

    print("\nBasePack build complete.")


if __name__ == "__main__":
    main()
