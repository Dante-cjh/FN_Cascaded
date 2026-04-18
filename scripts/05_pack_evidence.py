"""
05_pack_evidence.py
Heuristic evidence packer.

For each event in the test split:
  - If small model confidence >= threshold -> marked high-conf (will NOT be sent to LLM)
  - If small model confidence < threshold  -> pack evidence for LLM routing

Evidence selection strategy (heuristic, no training):
  1. source_text
  2. earliest 3 replies (chronological)
  3. longest 3 replies (by char length)
  4. one reply from each of the first 2 distinct branches
  -> deduplicate, cap at args.max_replies

Output: outputs/packed_events/packed_{threshold}.jsonl
Each line:
{
  "event_id": str,
  "label": int,
  "source_text": str,
  "selected_replies": [str, ...],
  "propagation_summary": {"num_replies": int, "max_depth": int, "num_branches": int},
  "small_model": {"pred": int, "confidence": float},
  "route_to_llm": bool
}
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_branch_map(structure: dict) -> dict[str, str]:
    """Map tweet_id -> branch_root (first child of source)."""
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


def select_replies(replies: list[dict], structure: dict, max_replies: int) -> list[str]:
    if not replies:
        return []

    # Sort by time for chronological order
    timed = sorted(replies, key=lambda r: r.get("time", ""))

    selected_ids = set()
    selected_texts = []

    def add(r):
        if r["tweet_id"] not in selected_ids and r["text"].strip():
            selected_ids.add(r["tweet_id"])
            selected_texts.append(r["text"].strip())

    # 1. Earliest 3
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


def pack_events(args):
    events = load_jsonl(Path(args.events))
    preds = load_jsonl(Path(args.predictions))

    pred_map = {p["event_id"]: p for p in preds}
    threshold = args.threshold

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    routed = 0
    with open(output_path, "w") as out:
        for e in events:
            eid = e["event_id"]
            p = pred_map.get(eid)
            if p is None:
                print(f"  [WARN] No prediction for event {eid}, skipping")
                continue

            route_to_llm = p["confidence"] < threshold
            selected = []
            if route_to_llm:
                selected = select_replies(
                    e.get("replies", []),
                    e.get("structure", {}),
                    args.max_replies,
                )

            record = {
                "event_id": eid,
                "label": e["label"],
                "source_text": e["source_text"],
                "selected_replies": selected,
                "propagation_summary": {
                    "num_replies": e["meta"]["num_replies"],
                    "max_depth": e["meta"]["max_depth"],
                    "num_branches": e["meta"]["num_branches"],
                },
                "small_model": {
                    "pred": p["pred"],
                    "confidence": p["confidence"],
                },
                "route_to_llm": route_to_llm,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            if route_to_llm:
                routed += 1

    llm_rate = routed / total if total else 0
    print(f"Packed {total} events -> {output_path}")
    print(f"  Threshold:       {threshold}")
    print(f"  Routed to LLM:   {routed}/{total} ({llm_rate:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="data/processed/test.jsonl")
    parser.add_argument("--predictions", default="outputs/small_model/test_predictions.jsonl")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--max_replies", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        thr_str = str(args.threshold).replace(".", "")
        args.output = f"outputs/packed_events/packed_{thr_str}.jsonl"

    pack_events(args)


if __name__ == "__main__":
    main()
