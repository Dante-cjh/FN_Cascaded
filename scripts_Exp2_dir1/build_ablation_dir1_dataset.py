"""
scripts_Exp2_dir1/build_ablation_dir1_dataset.py
Build leave-one-out ablation datasets for Exp-2 Direction 1 (Verifiability).

Strategy: start from the full 7-field LLM augmentation, remove ONE field at a time.

Reads (already generated):
  data/processed/basepack_{split}.jsonl
  outputs/exp2_dir1/llm_aug_{split}.jsonl

Writes:
  data/processed/exp2_ablation_dir1/{ablation_mode}_{split}.jsonl

ablation_mode     fields INCLUDED
  full            : all 7 fields
  wo_claim_core        : remove claim_core
  wo_claim_components  : remove claim_components
  wo_evidence_basis    : remove evidence_basis
  wo_verification_gaps : remove verification_gaps
  wo_source_grounding  : remove source_grounding
  wo_ambiguity_flags   : remove ambiguity_flags
  wo_followup_checks   : remove followup_checks

Run from project root:
  python scripts_Exp2_dir1/build_ablation_dir1_dataset.py --ablation_mode full
  python scripts_Exp2_dir1/build_ablation_dir1_dataset.py --ablation_mode wo_claim_core
  ... (etc.)
"""

import json
import argparse
from pathlib import Path

ALL_FIELDS = [
    "claim_core",
    "claim_components",
    "evidence_basis",
    "verification_gaps",
    "source_grounding",
    "ambiguity_flags",
    "followup_checks",
]

FIELD_SETS: dict[str, list[str]] = {
    "full":                  ALL_FIELDS,
    "wo_claim_core":         [f for f in ALL_FIELDS if f != "claim_core"],
    "wo_claim_components":   [f for f in ALL_FIELDS if f != "claim_components"],
    "wo_evidence_basis":     [f for f in ALL_FIELDS if f != "evidence_basis"],
    "wo_verification_gaps":  [f for f in ALL_FIELDS if f != "verification_gaps"],
    "wo_source_grounding":   [f for f in ALL_FIELDS if f != "source_grounding"],
    "wo_ambiguity_flags":    [f for f in ALL_FIELDS if f != "ambiguity_flags"],
    "wo_followup_checks":    [f for f in ALL_FIELDS if f != "followup_checks"],
}

REMOVED_FIELD: dict[str, str] = {
    "full":                  "(none — all 7 fields included)",
    "wo_claim_core":         "claim_core",
    "wo_claim_components":   "claim_components",
    "wo_evidence_basis":     "evidence_basis",
    "wo_verification_gaps":  "verification_gaps",
    "wo_source_grounding":   "source_grounding",
    "wo_ambiguity_flags":    "ambiguity_flags",
    "wo_followup_checks":    "followup_checks",
}

# Which fields are strings vs. lists (matches run_llm_preprocess_dir1.py schema)
STRING_FIELDS = {"claim_core", "evidence_basis", "source_grounding"}
LIST_FIELDS   = {"claim_components", "verification_gaps", "ambiguity_flags", "followup_checks"}


# --------------------------------------------------------------------------- #
#  Formatter
# --------------------------------------------------------------------------- #

def format_aug_block(aug: dict, fields: list[str]) -> str:
    lines = ["[LLM_AUGMENTATION]"]

    if "claim_core" in fields:
        v = aug.get("claim_core", "")
        if v:
            lines.append(f"claim_core: {v}")

    for field in ["claim_components", "ambiguity_flags", "verification_gaps", "followup_checks"]:
        if field in fields:
            lst = aug.get(field) or []
            if lst:
                lines.append(f"{field}:")
                for item in lst:
                    lines.append(f"  - {item}")

    for field in ["evidence_basis", "source_grounding"]:
        if field in fields:
            v = aug.get(field, "")
            if v:
                lines.append(f"{field}: {v}")

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
        description="Build leave-one-out ablation datasets for Exp-2 Dir1 (Verifiability)"
    )
    parser.add_argument(
        "--ablation_mode", required=True,
        choices=list(FIELD_SETS.keys()),
    )
    parser.add_argument("--basepack_dir", default="data/processed")
    parser.add_argument("--llm_aug_dir",  default="outputs/exp2_dir1",
                        help="Dir containing llm_aug_{split}.jsonl (already generated)")
    parser.add_argument("--output_dir",   default="data/processed/exp2_ablation_dir1")
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
