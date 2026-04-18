"""
06_build_augmented_dataset.py
Exp-2: Merge BasePack + LLM_Aug into the augmented dataset for small model training.

Reads:
  data/processed/basepack_{split}.jsonl
  outputs/exp2_llm_pre/llm_aug_{split}.jsonl

Output: data/processed/augmented_{split}.jsonl
Each line:
{
  "event_id":       str,
  "label":          int,
  "augmented_text": str,    # [ORIGINAL_EVENT] BasePack + [LLM_AUGMENTATION] block
  "basepack_text":  str,    # kept for reference / fallback
  "llm_aug":        dict    # the parsed LLM augmentation (or null if failed)
}

Events where LLM augmentation failed are included with augmented_text = basepack_text
(graceful fallback so training is never blocked by partial LLM failures).

This script supersedes 06_build_augmented_input.py with a cleaner interface
that aligns with the input_mode="base_plus_llm_aug" convention used in train.py.
"""

import json
import argparse
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Augmented text formatter
# --------------------------------------------------------------------------- #

def format_llm_aug_block(aug: dict) -> str:
    """Render the LLM augmentation dict as a structured text block."""
    lines = ["[LLM_AUGMENTATION]"]

    claim = aug.get("claim_summary", "")
    if claim:
        lines.append(f"claim_summary: {claim}")

    supporting = aug.get("supporting_signals") or []
    if supporting:
        lines.append("supporting_signals:")
        for s in supporting:
            lines.append(f"  - {s}")

    refuting = aug.get("refuting_signals") or []
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
    """Combine BasePack and LLM augmentation into the final model input string."""
    if not aug:
        return basepack_text    # graceful fallback
    aug_block = format_llm_aug_block(aug)
    return f"[ORIGINAL_EVENT]\n{basepack_text}\n\n{aug_block}"


# --------------------------------------------------------------------------- #
#  Per-split processing
# --------------------------------------------------------------------------- #

def build_augmented_split(basepack_path: Path, llm_aug_path: Path, output_path: Path):
    with open(basepack_path) as f:
        basepacks = {json.loads(l)["event_id"]: json.loads(l) for l in f if l.strip()}

    llm_augs: dict[str, dict] = {}
    fallback_count = 0
    if llm_aug_path.exists():
        with open(llm_aug_path) as f:
            for line in f:
                if line.strip():
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
          f"({written} events, {fallback_count} fallback-to-basepack, "
          f"aug_rate={aug_rate:.1%})")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Exp-2: Build augmented dataset (BasePack + LLM_Aug)"
    )
    parser.add_argument(
        "--basepack_dir", default="data/processed",
        help="Directory containing basepack_{split}.jsonl",
    )
    parser.add_argument(
        "--llm_aug_dir", default="outputs/exp2_llm_pre",
        help="Directory containing llm_aug_{split}.jsonl",
    )
    parser.add_argument(
        "--output_dir", default="data/processed",
        help="Directory for augmented_{split}.jsonl outputs",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
    )
    args = parser.parse_args()

    for split in args.splits:
        bp_path  = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        aug_path = Path(args.llm_aug_dir)  / f"llm_aug_{split}.jsonl"
        out_path = Path(args.output_dir)   / f"augmented_{split}.jsonl"

        if not bp_path.exists():
            print(f"[WARN] Not found: {bp_path}, skipping.")
            continue

        print(f"Building augmented_{split}...")
        build_augmented_split(bp_path, aug_path, out_path)

    print("\nAugmented dataset build complete.")
    print("Next: bash scripts/06a_train_exp2.sh")


if __name__ == "__main__":
    main()
