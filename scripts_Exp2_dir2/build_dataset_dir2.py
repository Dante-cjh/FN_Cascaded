"""
scripts_Exp2_dir2/build_dataset_dir2.py
Direction 2 (Narrative/Manipulation): Merge BasePack + Dir2 LLM_Aug.

Fields formatted into augmented_text:
  narrative_frame, persuasion_cues, engagement_pattern, coordination_signals,
  evidence_visibility, attention_triggers, manipulation_risk_profile

Reads:
  data/processed/basepack_{split}.jsonl
  outputs/exp2_dir2/llm_aug_{split}.jsonl

Writes:
  data/processed/exp2_dir2/augmented_{split}.jsonl

Run from project root:
  python scripts_Exp2_dir2/build_dataset_dir2.py
"""

import json
import argparse
from pathlib import Path


def format_llm_aug_block(aug: dict) -> str:
    lines = ["[LLM_AUGMENTATION]"]

    for field, label in [
        ("narrative_frame",          "narrative_frame"),
        ("engagement_pattern",       "engagement_pattern"),
        ("evidence_visibility",      "evidence_visibility"),
        ("manipulation_risk_profile","manipulation_risk_profile"),
    ]:
        v = aug.get(field, "")
        if v:
            lines.append(f"{label}: {v}")

    for field, label in [
        ("persuasion_cues",       "persuasion_cues"),
        ("coordination_signals",  "coordination_signals"),
        ("attention_triggers",    "attention_triggers"),
    ]:
        lst = aug.get(field) or []
        if lst:
            lines.append(f"{label}:")
            for item in lst:
                lines.append(f"  - {item}")

    return "\n".join(lines)


def build_augmented_text(basepack_text: str, aug: dict | None) -> str:
    if not aug:
        return basepack_text
    block = format_llm_aug_block(aug)
    if block.strip() == "[LLM_AUGMENTATION]":
        return basepack_text
    return f"[ORIGINAL_EVENT]\n{basepack_text}\n\n{block}"


def build_split(basepack_path: Path, llm_aug_path: Path, output_path: Path):
    with open(basepack_path) as f:
        basepacks = {json.loads(l)["event_id"]: json.loads(l) for l in f if l.strip()}

    llm_augs: dict[str, dict] = {}
    if llm_aug_path.exists():
        with open(llm_aug_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    llm_augs[rec["event_id"]] = rec
    else:
        print(f"  [WARN] Not found: {llm_aug_path}. Using BasePack fallback.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = fallback_count = 0

    with open(output_path, "w") as fout:
        for eid, bp in basepacks.items():
            llm_rec = llm_augs.get(eid)
            aug = llm_rec["llm_aug"] if (llm_rec and llm_rec.get("parse_success")) else None
            if aug is None:
                fallback_count += 1
            augmented_text = build_augmented_text(bp["basepack_text"], aug)
            fout.write(json.dumps({
                "event_id":       eid,
                "label":          bp["label"],
                "augmented_text": augmented_text,
                "basepack_text":  bp["basepack_text"],
                "llm_aug":        aug,
            }, ensure_ascii=False) + "\n")
            written += 1

    aug_rate = (written - fallback_count) / max(written, 1)
    print(f"  -> {output_path}  "
          f"({written} events, {fallback_count} fallback, aug_rate={aug_rate:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Dir2 (Narrative/Manipulation): Build augmented dataset"
    )
    parser.add_argument("--basepack_dir", default="data/processed")
    parser.add_argument("--llm_aug_dir",  default="outputs/exp2_dir2")
    parser.add_argument("--output_dir",   default="data/processed/exp2_dir2")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    for split in args.splits:
        bp_path  = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        aug_path = Path(args.llm_aug_dir)  / f"llm_aug_{split}.jsonl"
        out_path = Path(args.output_dir)   / f"augmented_{split}.jsonl"
        if not bp_path.exists():
            print(f"[WARN] Not found: {bp_path}, skipping.")
            continue
        print(f"Building {split}...")
        build_split(bp_path, aug_path, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
