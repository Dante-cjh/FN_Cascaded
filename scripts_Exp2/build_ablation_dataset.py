"""
scripts_Exp2/build_ablation_dataset.py
Build partial-augmented datasets for Exp-2 ablation study.

Reads (already generated, not re-generated):
  data/processed/basepack_{split}.jsonl
  outputs/exp2_llm_pre/llm_aug_{split}.jsonl

Writes:
  data/processed/exp2_ablation/{ablation_mode}_{split}.jsonl

ablation_mode → fields included in [LLM_AUGMENTATION] block
  claim_only      : claim_summary
  claim_signals   : claim_summary + supporting_signals + refuting_signals
  signals_only    : supporting_signals + refuting_signals
  analysis_only   : conflict_summary + risk_note
  claim_analysis  : claim_summary + conflict_summary + risk_note

abl1 (basepack_only) and abl7 (full) reuse existing files; no need to run this
script for those two conditions.

Run from project root:
  python scripts_Exp2/build_ablation_dataset.py --ablation_mode claim_only
  python scripts_Exp2/build_ablation_dataset.py --ablation_mode claim_signals
  python scripts_Exp2/build_ablation_dataset.py --ablation_mode signals_only
  python scripts_Exp2/build_ablation_dataset.py --ablation_mode analysis_only
  python scripts_Exp2/build_ablation_dataset.py --ablation_mode claim_analysis
"""

import json
import argparse
from pathlib import Path

# Ordered field sets per ablation mode
FIELD_SETS: dict[str, list[str]] = {
    "claim_only":     ["claim_summary"],
    "claim_signals":  ["claim_summary", "supporting_signals", "refuting_signals"],
    "signals_only":   ["supporting_signals", "refuting_signals"],
    "analysis_only":  ["conflict_summary", "risk_note"],
    "claim_analysis": ["claim_summary", "conflict_summary", "risk_note"],
}


# --------------------------------------------------------------------------- #
#  Formatter
# --------------------------------------------------------------------------- #

def format_partial_aug_block(aug: dict, fields: list[str]) -> str:
    lines = ["[LLM_AUGMENTATION]"]

    if "claim_summary" in fields:
        claim = aug.get("claim_summary", "")
        if claim:
            lines.append(f"claim_summary: {claim}")

    if "supporting_signals" in fields:
        supporting = aug.get("supporting_signals") or []
        if supporting:
            lines.append("supporting_signals:")
            for s in supporting:
                lines.append(f"  - {s}")

    if "refuting_signals" in fields:
        refuting = aug.get("refuting_signals") or []
        if refuting:
            lines.append("refuting_signals:")
            for r in refuting:
                lines.append(f"  - {r}")

    if "conflict_summary" in fields:
        conflict = aug.get("conflict_summary", "")
        if conflict:
            lines.append(f"conflict_summary: {conflict}")

    if "risk_note" in fields:
        risk = aug.get("risk_note", "")
        if risk:
            lines.append(f"risk_note: {risk}")

    return "\n".join(lines)


def build_augmented_text(basepack_text: str, aug: dict | None, fields: list[str]) -> str:
    if not aug:
        return basepack_text
    block = format_partial_aug_block(aug, fields)
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
        print(f"  [WARN] LLM aug file not found: {llm_aug_path}. Falling back to BasePack.")

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
        description="Build partial-augmented datasets for Exp-2 ablation"
    )
    parser.add_argument(
        "--ablation_mode", required=True,
        choices=list(FIELD_SETS.keys()),
        help="Which LLM fields to include in the augmented_text block",
    )
    parser.add_argument("--basepack_dir", default="data/processed",
                        help="Dir with basepack_{split}.jsonl")
    parser.add_argument("--llm_aug_dir",  default="outputs/exp2_llm_pre",
                        help="Dir with llm_aug_{split}.jsonl (already generated)")
    parser.add_argument("--output_dir",   default="data/processed/exp2_ablation",
                        help="Output dir for partial-augmented files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    fields = FIELD_SETS[args.ablation_mode]
    print(f"ablation_mode : {args.ablation_mode}")
    print(f"fields        : {fields}")
    print()

    for split in args.splits:
        bp_path  = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        aug_path = Path(args.llm_aug_dir)  / f"llm_aug_{split}.jsonl"
        out_path = Path(args.output_dir)   / f"{args.ablation_mode}_{split}.jsonl"

        if not bp_path.exists():
            print(f"[WARN] Not found: {bp_path}, skipping.")
            continue

        print(f"Building {split}...")
        build_split(bp_path, aug_path, out_path, fields)

    print("\nDone. Next: run the corresponding train script in scripts_Exp2/.")


if __name__ == "__main__":
    main()
