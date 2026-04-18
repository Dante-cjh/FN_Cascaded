"""
01_build_events.py
Traverse the extracted PHEME directory and build a flat events.jsonl file.

Expected raw data layout (after extracting PHEME_veracity.tar.bz2):
  data/raw/PHEME/all-rnr-annotated-threads/
    {topic}-all-rnr-threads/
      rumours/
        {tweet_id}/
          source-tweets/{tweet_id}.json
          reactions/*.json
          structure.json
          annotation.json
      non-rumours/
        {tweet_id}/
          ...

Output: data/processed/events.jsonl
Each line is a JSON object:
{
  "event_id":   str,
  "topic":      str,
  "label":      int,   # 1=rumor, 0=non-rumor
  "source_text": str,
  "replies": [
    {"tweet_id": str, "text": str, "parent": str|null, "time": str}
  ],
  "structure":  dict,
  "meta": {
    "num_replies": int,
    "max_depth":   int,
    "num_branches": int
  }
}
"""

import os
import json
import argparse
from pathlib import Path


def compute_tree_stats(structure: dict) -> dict:
    """Compute max_depth and num_branches from structure.json."""
    if not structure:
        return {"max_depth": 0, "num_branches": 0}

    def _depth(node, d=0):
        if not node:
            return d
        return max(_depth(v, d + 1) for v in node.values()) if isinstance(node, dict) else d

    def _branches(node):
        if not isinstance(node, dict) or not node:
            return 0
        return (1 if len(node) > 1 else 0) + sum(_branches(v) for v in node.values())

    return {
        "max_depth": _depth(structure),
        "num_branches": _branches(structure),
    }


def load_event(event_dir: Path, label: int, topic: str) -> dict | None:
    event_id = event_dir.name

    # Source tweet
    src_dir = event_dir / "source-tweets"
    # 过滤掉 macOS 资源叉文件（._xxx.json）
    src_files = [p for p in src_dir.glob("*.json") if not p.name.startswith("._")] if src_dir.exists() else []
    if not src_files:
        print(f"  [SKIP] No source tweet in {event_dir}")
        return None
    with open(src_files[0], encoding="utf-8", errors="replace") as f:
        try:
            src_json = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  [SKIP] Invalid source tweet JSON in {event_dir}")
            return None
    source_text = src_json.get("text", "")

    # Reactions
    replies = []
    reactions_dir = event_dir / "reactions"
    if reactions_dir.exists():
        # 过滤掉 macOS 资源叉文件（._xxx.json）
        for rfile in sorted(p for p in reactions_dir.glob("*.json") if not p.name.startswith("._")):
            with open(rfile, encoding="utf-8", errors="replace") as f:
                try:
                    rdata = json.load(f)
                except json.JSONDecodeError:
                    continue
            replies.append({
                "tweet_id": str(rdata.get("id_str", rfile.stem)),
                "text": rdata.get("text", ""),
                "parent": str(rdata.get("in_reply_to_status_id_str", "")),
                "time": rdata.get("created_at", ""),
            })

    # Structure
    structure_path = event_dir / "structure.json"
    structure = {}
    if structure_path.exists():
        with open(structure_path, encoding="utf-8", errors="replace") as f:
            try:
                structure = json.load(f)
            except json.JSONDecodeError:
                pass

    stats = compute_tree_stats(structure)
    stats["num_replies"] = len(replies)

    return {
        "event_id": event_id,
        "topic": topic,
        "label": label,
        "source_text": source_text,
        "replies": replies,
        "structure": structure,
        "meta": stats,
    }


def build_events(raw_root: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0

    with open(output_path, "w") as out:
        for topic_dir in sorted(raw_root.iterdir()):
            if not topic_dir.is_dir() or topic_dir.name.startswith("."):
                continue
            topic = topic_dir.name.replace("-all-rnr-threads", "")
            print(f"Processing topic: {topic}")

            for split_name, label in [("rumours", 1), ("non-rumours", 0)]:
                split_dir = topic_dir / split_name
                if not split_dir.exists():
                    continue
                for event_dir in sorted(split_dir.iterdir()):
                    if not event_dir.is_dir() or event_dir.name.startswith("."):
                        continue
                    event = load_event(event_dir, label, topic)
                    if event is None:
                        skipped += 1
                        continue
                    out.write(json.dumps(event, ensure_ascii=False) + "\n")
                    total += 1

    print(f"\nDone. Written {total} events ({skipped} skipped) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build events.jsonl from raw PHEME")
    parser.add_argument("--raw_root", default="data/raw/PHEME/all-rnr-annotated-threads")
    parser.add_argument("--output", default="data/processed/events.jsonl")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        print(f"ERROR: raw_root not found: {raw_root}")
        print("Please extract PHEME_veracity.tar.bz2 first:")
        print("  tar -xzf PHEME/PHEME_veracity.tar.bz2 -C data/raw/PHEME/")
        return

    build_events(raw_root, Path(args.output))


if __name__ == "__main__":
    main()
