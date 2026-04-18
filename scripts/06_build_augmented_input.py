"""
06_build_augmented_input.py
Exp-2: Merge BasePack + LLM_Aug into augmented input for small model training.

Reads:
  data/processed/basepack_{split}.jsonl
  outputs/exp2_llm_pre/llm_aug_{split}.jsonl

Output: data/processed/augmented_{split}.jsonl
Each line:
{
  "event_id":       str,
  "label":          int,
  "augmented_text": str,    # BasePack + LLM_Aug concatenated
  "basepack_text":  str,    # original BasePack (for reference)
  "llm_aug":        dict    # the LLM augmentation block
}

Events where LLM augmentation failed (parse_success=False) are included
with augmented_text = basepack_text only (graceful fallback).
"""

import json
import argparse
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Augmented text formatter
# --------------------------------------------------------------------------- #

def format_llm_aug(aug: dict) -> str:
    """Format the LLM augmentation dict as a structured text block."""
    lines = ["[LLM_AUGMENTATION]"]

    claim = aug.get("claim_summary", "")
    if claim:
        lines.append(f"claim_summary: {claim}")

    supporting = aug.get("supporting_signals", [])
    if supporting:
        lines.append("supporting_signals:")
        for s in supporting:
            lines.append(f"  - {s}")

    refuting = aug.get("refuting_signals", [])
    if refuting:
        lines.append("refuting_signals:")
        for r in refuting:
            lines.append(f"  - {r}")

    conflict = aug.get("conflict_summary", "")
    if conflict:
        lines.append(f"conflict_summary: {conflict}")

    risk = aug.get("risk_note", "")
    if risk:
        lines.append(f"risk_note: {risk}")

    return "\n".join(lines)


def build_augmented_text(basepack_text: str, aug: dict | None) -> str:
    if not aug:
        return basepack_text
    aug_block = format_llm_aug(aug)
    return f"[ORIGINAL_EVENT]\n{basepack_text}\n\n{aug_block}"


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def build_augmented_split(basepack_path: Path, llm_aug_path: Path, output_path: Path):
    with open(basepack_path) as f:
        basepacks = {json.loads(line)["event_id"]: json.loads(line)
                     for line in open(basepack_path) if line.strip()}

    llm_augs: dict[str, dict] = {}
    fallback_count = 0
    if llm_aug_path.exists():
        for line in open(llm_aug_path):
            if not line.strip():
                continue
            rec = json.loads(line)
            llm_augs[rec["event_id"]] = rec
    else:
        print(f"  [WARN] LLM aug file not found: {llm_aug_path}. Using BasePack only.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as fout:
        for eid, bp in basepacks.items():
            llm_rec = llm_augs.get(eid)
            aug = llm_rec["llm_aug"] if (llm_rec and llm_rec.get("parse_success")) else None
            if aug is None:
                fallback_count += 1

            augmented_text = build_augmented_text(bp["basepack_text"], aug)
            record = {
                "event_id": eid,
                "label": bp["label"],
                "augmented_text": augmented_text,
                "basepack_text": bp["basepack_text"],
                "llm_aug": aug,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"  -> {output_path}  ({written} events, {fallback_count} fallback-to-basepack)")


def main():
    parser = argparse.ArgumentParser(description="Exp-2: Build augmented input files")
    parser.add_argument("--basepack_dir", default="data/processed",
                        help="Directory containing basepack_{split}.jsonl")
    parser.add_argument("--llm_aug_dir", default="outputs/exp2_llm_pre",
                        help="Directory containing llm_aug_{split}.jsonl")
    parser.add_argument("--output_dir", default="data/processed",
                        help="Directory for augmented_{split}.jsonl outputs")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    for split in args.splits:
        bp_path = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        aug_path = Path(args.llm_aug_dir) / f"llm_aug_{split}.jsonl"
        out_path = Path(args.output_dir) / f"augmented_{split}.jsonl"

        if not bp_path.exists():
            print(f"[WARN] Not found: {bp_path}, skipping.")
            continue

        print(f"Building augmented_{split}...")
        build_augmented_split(bp_path, aug_path, out_path)

    print("\nAugmented input build complete.")


if __name__ == "__main__":
    main()
