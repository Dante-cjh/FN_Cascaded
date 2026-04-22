"""
scripts_Exp2_v2/build_ablation_v2_dataset.py
Build leave-one-out ablation datasets for Exp-2 (v2).

Strategy: start from the full 5-field LLM augmentation, remove ONE field at a time.
This tells us how much each individual field contributes to the full model.

Reads (already generated, not re-generated):
  data/processed/basepack_{split}.jsonl
  outputs/exp2_llm_pre/llm_aug_{split}.jsonl

Writes:
  data/processed/exp2_ablation_v2/{ablation_mode}_{split}.jsonl

ablation_mode   fields INCLUDED (one field is left out each time)
  full          : claim_summary + supporting_signals + refuting_signals + conflict_summary + risk_note
  wo_claim      : supporting_signals + refuting_signals + conflict_summary + risk_note
  wo_supporting : claim_summary + refuting_signals + conflict_summary + risk_note
  wo_refuting   : claim_summary + supporting_signals + conflict_summary + risk_note
  wo_conflict   : claim_summary + supporting_signals + refuting_signals + risk_note
  wo_risk       : claim_summary + supporting_signals + refuting_signals + conflict_summary

Run from project root:
  python scripts_Exp2_v2/build_ablation_v2_dataset.py --ablation_mode full
  python scripts_Exp2_v2/build_ablation_v2_dataset.py --ablation_mode wo_claim
  ... (etc.)
"""

import json
import argparse
from pathlib import Path

ALL_FIELDS = [
    "claim_summary",
    "supporting_signals",
    "refuting_signals",
    "conflict_summary",
    "risk_note",
]

# Leave-one-out field sets
FIELD_SETS: dict[str, list[str]] = {
    "full":          ALL_FIELDS,
    "wo_claim":      [f for f in ALL_FIELDS if f != "claim_summary"],
    "wo_supporting": [f for f in ALL_FIELDS if f != "supporting_signals"],
    "wo_refuting":   [f for f in ALL_FIELDS if f != "refuting_signals"],
    "wo_conflict":   [f for f in ALL_FIELDS if f != "conflict_summary"],
    "wo_risk":       [f for f in ALL_FIELDS if f != "risk_note"],
}

REMOVED_FIELD: dict[str, str] = {
    "full":          "(none — all 5 fields included)",
    "wo_claim":      "claim_summary",
    "wo_supporting": "supporting_signals",
    "wo_refuting":   "refuting_signals",
    "wo_conflict":   "conflict_summary",
    "wo_risk":       "risk_note",
}


# --------------------------------------------------------------------------- #
#  Formatter
# --------------------------------------------------------------------------- #

def format_aug_block(aug: dict, fields: list[str]) -> str:
    lines = ["[LLM_AUGMENTATION]"]

    if "claim_summary" in fields:
        v = aug.get("claim_summary", "")
        if v:
            lines.append(f"claim_summary: {v}")

    if "supporting_signals" in fields:
        lst = aug.get("supporting_signals") or []
        if lst:
            lines.append("supporting_signals:")
            for s in lst:
                lines.append(f"  - {s}")

    if "refuting_signals" in fields:
        lst = aug.get("refuting_signals") or []
        if lst:
            lines.append("refuting_signals:")
            for r in lst:
                lines.append(f"  - {r}")

    if "conflict_summary" in fields:
        v = aug.get("conflict_summary", "")
        if v:
            lines.append(f"conflict_summary: {v}")

    if "risk_note" in fields:
        v = aug.get("risk_note", "")
        if v:
            lines.append(f"risk_note: {v}")

    return "\n".join(lines)


def build_augmented_text(basepack_text: str, aug: dict | None, fields: list[str]) -> str:
    if not aug:
        return basepack_text
    block = format_aug_block(aug, fields)
    if block.strip() == "[LLM_AUGMENTATION]":
        return basepack_text
    return f"[ORIGINAL_EVENT]\n{basepack_text}\n\n{block}"


# --------------------------------------------------------------------------- #
#  Per-split processing
# --------------------------------------------------------------------------- #

def build_split(
    basepack_path: Path,
    llm_aug_path: Path,
    output_path: Path,
    fields: list[str],
):
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
        print(f"  [WARN] Not found: {llm_aug_path}. Falling back to BasePack only.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = fallback_count = 0

    with open(output_path, "w") as fout:
        for eid, bp in basepacks.items():
            llm_rec = llm_augs.get(eid)
            aug = llm_rec["llm_aug"] if (llm_rec and llm_rec.get("parse_success")) else None
            if aug is None:
                fallback_count += 1

            augmented_text = build_augmented_text(bp["basepack_text"], aug, fields)
            record = {
                "event_id":       eid,
                "label":          bp["label"],
                "augmented_text": augmented_text,
                "basepack_text":  bp["basepack_text"],
                "llm_aug":        aug,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    aug_rate = (written - fallback_count) / max(written, 1)
    print(f"  -> {output_path}  "
          f"({written} events, {fallback_count} fallback, aug_rate={aug_rate:.1%})")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Build leave-one-out ablation datasets for Exp-2 v2"
    )
    parser.add_argument(
        "--ablation_mode", required=True,
        choices=list(FIELD_SETS.keys()),
        help=(
            "full          → all 5 fields\n"
            "wo_claim      → remove claim_summary\n"
            "wo_supporting → remove supporting_signals\n"
            "wo_refuting   → remove refuting_signals\n"
            "wo_conflict   → remove conflict_summary\n"
            "wo_risk       → remove risk_note"
        ),
    )
    parser.add_argument("--basepack_dir", default="data/processed",
                        help="Dir containing basepack_{split}.jsonl")
    parser.add_argument("--llm_aug_dir",  default="outputs/exp2_llm_pre",
                        help="Dir containing llm_aug_{split}.jsonl (already generated)")
    parser.add_argument("--output_dir",   default="data/processed/exp2_ablation_v2",
                        help="Output dir for leave-one-out augmented files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    fields  = FIELD_SETS[args.ablation_mode]
    removed = REMOVED_FIELD[args.ablation_mode]

    print(f"ablation_mode : {args.ablation_mode}")
    print(f"field removed : {removed}")
    print(f"fields used   : {fields}")
    print()

    for split in args.splits:
        bp_path  = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        aug_path = Path(args.llm_aug_dir)  / f"llm_aug_{split}.jsonl"
        out_path = Path(args.output_dir)   / f"{args.ablation_mode}_{split}.jsonl"

        if not bp_path.exists():
            print(f"[WARN] Not found: {bp_path}, skipping.")
            continue

        print(f"  Building {split}...")
        build_split(bp_path, aug_path, out_path, fields)

    print("\nDone.")


if __name__ == "__main__":
    main()
