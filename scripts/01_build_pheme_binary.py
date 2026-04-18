"""
01_build_pheme_binary.py
Build a binary Fake/True events file from PHEME veracity annotations.

Only processes rumour threads (non-rumours have no veracity annotation).
Drops 'unverified' events.

Label mapping:
  misinformation=0, true=1  → True  → label 0
  misinformation=1, true=0  → Fake  → label 1

Output: data/processed/binary_events.jsonl
Each line:
{
  "event_id":    str,
  "topic":       str,
  "veracity":    str,   # "true" or "false"
  "label":       int,   # 0=True, 1=Fake
  "source_text": str,
  "replies": [{"tweet_id": str, "text": str, "parent": str, "time": str}],
  "structure":   dict,
  "meta": {"num_replies": int, "max_depth": int, "num_branches": int, "time_span": str}
}
"""

import json
import argparse
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Veracity annotation logic (from PHEME/convert_veracity_annotations.py)
# --------------------------------------------------------------------------- #

def convert_annotation(annotation: dict) -> str | None:
    """Return 'true', 'false', 'unverified', or None if annotation is invalid."""
    has_misinfo = "misinformation" in annotation
    has_true = "true" in annotation

    if has_misinfo and has_true:
        m = int(annotation["misinformation"])
        t = int(annotation["true"])
        if m == 0 and t == 0:
            return "unverified"
        elif m == 0 and t == 1:
            return "true"
        elif m == 1 and t == 0:
            return "false"
        else:
            return None
    elif has_misinfo and not has_true:
        m = int(annotation["misinformation"])
        return "false" if m == 1 else "unverified"
    else:
        return None


# --------------------------------------------------------------------------- #
#  Tree stats (reused from 01_build_events.py)
# --------------------------------------------------------------------------- #

def compute_tree_stats(structure: dict) -> dict:
    if not structure:
        return {"max_depth": 0, "num_branches": 0}

    def _depth(node, d=0):
        if not node or not isinstance(node, dict):
            return d
        return max(_depth(v, d + 1) for v in node.values())

    def _branches(node):
        if not isinstance(node, dict) or not node:
            return 0
        return (1 if len(node) > 1 else 0) + sum(_branches(v) for v in node.values())

    return {
        "max_depth": _depth(structure),
        "num_branches": _branches(structure),
    }


def compute_time_span(replies: list[dict]) -> str:
    """Return rough time span string, e.g. '2h30m'."""
    times = [r["time"] for r in replies if r.get("time")]
    if len(times) < 2:
        return "unknown"
    times_sorted = sorted(times)
    return f"{times_sorted[0]} ~ {times_sorted[-1]}"


# --------------------------------------------------------------------------- #
#  Event loader
# --------------------------------------------------------------------------- #

def load_rumour_event(event_dir: Path, topic: str) -> dict | None:
    event_id = event_dir.name

    # Annotation (veracity)
    ann_path = event_dir / "annotation.json"
    if not ann_path.exists():
        return None
    with open(ann_path, encoding="utf-8", errors="replace") as f:
        try:
            ann = json.load(f)
        except json.JSONDecodeError:
            return None

    veracity = convert_annotation(ann)
    if veracity is None or veracity == "unverified":
        return None

    label = 1 if veracity == "false" else 0   # Fake=1, True=0

    # Source tweet
    src_dir = event_dir / "source-tweets"
    src_files = [p for p in src_dir.glob("*.json") if not p.name.startswith("._")] if src_dir.exists() else []
    if not src_files:
        return None
    with open(src_files[0], encoding="utf-8", errors="replace") as f:
        try:
            src_json = json.load(f)
        except json.JSONDecodeError:
            return None
    source_text = src_json.get("text", "").strip()

    # Replies
    replies = []
    reactions_dir = event_dir / "reactions"
    if reactions_dir.exists():
        for rfile in sorted(p for p in reactions_dir.glob("*.json") if not p.name.startswith("._")):
            with open(rfile, encoding="utf-8", errors="replace") as f:
                try:
                    rdata = json.load(f)
                except json.JSONDecodeError:
                    continue
            replies.append({
                "tweet_id": str(rdata.get("id_str", rfile.stem)),
                "text": rdata.get("text", "").strip(),
                "parent": str(rdata.get("in_reply_to_status_id_str", "")),
                "time": rdata.get("created_at", ""),
            })

    # Structure
    structure = {}
    struct_path = event_dir / "structure.json"
    if struct_path.exists():
        with open(struct_path, encoding="utf-8", errors="replace") as f:
            try:
                structure = json.load(f)
            except json.JSONDecodeError:
                pass

    stats = compute_tree_stats(structure)
    stats["num_replies"] = len(replies)
    stats["time_span"] = compute_time_span(replies)

    return {
        "event_id": event_id,
        "topic": topic,
        "veracity": veracity,
        "label": label,
        "source_text": source_text,
        "replies": replies,
        "structure": structure,
        "meta": stats,
    }


# --------------------------------------------------------------------------- #
#  Main builder
# --------------------------------------------------------------------------- #

def build_binary_events(raw_root: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped_missing_ann = 0
    skipped_unverified = 0

    label_counts = {"true": 0, "false": 0}

    with open(output_path, "w") as out:
        for topic_dir in sorted(raw_root.iterdir()):
            if not topic_dir.is_dir() or topic_dir.name.startswith("."):
                continue
            topic = topic_dir.name.replace("-all-rnr-threads", "")
            print(f"Processing topic: {topic}")

            rumours_dir = topic_dir / "rumours"
            if not rumours_dir.exists():
                continue

            for event_dir in sorted(rumours_dir.iterdir()):
                if not event_dir.is_dir() or event_dir.name.startswith("."):
                    continue

                ann_path = event_dir / "annotation.json"
                if not ann_path.exists():
                    skipped_missing_ann += 1
                    continue

                event = load_rumour_event(event_dir, topic)
                if event is None:
                    skipped_unverified += 1
                    continue

                out.write(json.dumps(event, ensure_ascii=False) + "\n")
                label_counts[event["veracity"]] += 1
                total += 1

    print(f"\nDone. Written {total} events -> {output_path}")
    print(f"  Fake (false): {label_counts['false']}")
    print(f"  True (true):  {label_counts['true']}")
    print(f"  Skipped (no annotation):  {skipped_missing_ann}")
    print(f"  Skipped (unverified):     {skipped_unverified}")


def main():
    parser = argparse.ArgumentParser(description="Build binary Fake/True PHEME dataset")
    parser.add_argument("--raw_root", default="data/raw/PHEME/all-rnr-annotated-threads")
    parser.add_argument("--output", default="data/processed/binary_events.jsonl")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        print(f"ERROR: raw_root not found: {raw_root}")
        return

    build_binary_events(raw_root, Path(args.output))


if __name__ == "__main__":
    main()
