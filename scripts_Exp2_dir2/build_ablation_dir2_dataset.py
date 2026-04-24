"""
scripts_Exp2_dir2/build_ablation_dir2_dataset.py
Build leave-one-out ablation datasets for Exp-2 Direction 2 (Narrative/Manipulation).

Strategy: start from the full 7-field LLM augmentation, remove ONE field at a time.

Reads (already generated):
  data/processed/basepack_{split}.jsonl
  outputs/exp2_dir2/llm_aug_{split}.jsonl

Writes:
  data/processed/exp2_ablation_dir2/{ablation_mode}_{split}.jsonl

ablation_mode                  fields INCLUDED
  full                         : all 7 fields
  wo_narrative_frame           : remove narrative_frame
  wo_persuasion_cues           : remove persuasion_cues
  wo_engagement_pattern        : remove engagement_pattern
  wo_coordination_signals      : remove coordination_signals
  wo_evidence_visibility       : remove evidence_visibility
  wo_attention_triggers        : remove attention_triggers
  wo_manipulation_risk_profile : remove manipulation_risk_profile

Run from project root:
  python scripts_Exp2_dir2/build_ablation_dir2_dataset.py --ablation_mode full
  python scripts_Exp2_dir2/build_ablation_dir2_dataset.py --ablation_mode wo_narrative_frame
  ... (etc.)
"""

import json
import argparse
from pathlib import Path

ALL_FIELDS = [
    "narrative_frame",
    "persuasion_cues",
    "engagement_pattern",
    "coordination_signals",
    "evidence_visibility",
    "attention_triggers",
    "manipulation_risk_profile",
]

FIELD_SETS: dict[str, list[str]] = {
    "full":                          ALL_FIELDS,
    "wo_narrative_frame":            [f for f in ALL_FIELDS if f != "narrative_frame"],
    "wo_persuasion_cues":            [f for f in ALL_FIELDS if f != "persuasion_cues"],
    "wo_engagement_pattern":         [f for f in ALL_FIELDS if f != "engagement_pattern"],
    "wo_coordination_signals":       [f for f in ALL_FIELDS if f != "coordination_signals"],
    "wo_evidence_visibility":        [f for f in ALL_FIELDS if f != "evidence_visibility"],
    "wo_attention_triggers":         [f for f in ALL_FIELDS if f != "attention_triggers"],
    "wo_manipulation_risk_profile":  [f for f in ALL_FIELDS if f != "manipulation_risk_profile"],
}

REMOVED_FIELD: dict[str, str] = {
    "full":                          "(none — all 7 fields included)",
    "wo_narrative_frame":            "narrative_frame",
    "wo_persuasion_cues":            "persuasion_cues",
    "wo_engagement_pattern":         "engagement_pattern",
    "wo_coordination_signals":       "coordination_signals",
    "wo_evidence_visibility":        "evidence_visibility",
    "wo_attention_triggers":         "attention_triggers",
    "wo_manipulation_risk_profile":  "manipulation_risk_profile",
}

# Which fields are strings vs. lists (matches run_llm_preprocess_dir2.py schema)
STRING_FIELDS = {"narrative_frame", "engagement_pattern", "evidence_visibility", "manipulation_risk_profile"}
LIST_FIELDS   = {"persuasion_cues", "coordination_signals", "attention_triggers"}


# --------------------------------------------------------------------------- #
#  Formatter
# --------------------------------------------------------------------------- #

def format_aug_block(aug: dict, fields: list[str]) -> str:
    lines = ["[LLM_AUGMENTATION]"]

    for field in ["narrative_frame", "engagement_pattern", "evidence_visibility", "manipulation_risk_profile"]:
        if field in fields:
            v = aug.get(field, "")
            if v:
                lines.append(f"{field}: {v}")

    for field in ["persuasion_cues", "coordination_signals", "attention_triggers"]:
        if field in fields:
            lst = aug.get(field) or []
            if lst:
                lines.append(f"{field}:")
                for item in lst:
                    lines.append(f"  - {item}")

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
        description="Build leave-one-out ablation datasets for Exp-2 Dir2 (Narrative/Manipulation)"
    )
    parser.add_argument(
        "--ablation_mode", required=True,
        choices=list(FIELD_SETS.keys()),
    )
    parser.add_argument("--basepack_dir", default="data/processed")
    parser.add_argument("--llm_aug_dir",  default="outputs/exp2_dir2",
                        help="Dir containing llm_aug_{split}.jsonl (already generated)")
    parser.add_argument("--output_dir",   default="data/processed/exp2_ablation_dir2")
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
