"""
09_case_study.py
Build case studies comparing the three experiments.

For each of the four interesting case categories, selects up to n_cases examples:
  A) Exp-1 correct, Exp-2 wrong
  B) Exp-1 wrong,   Exp-2 correct
  C) Exp-1 correct, Exp-3 wrong
  D) Exp-1 wrong,   Exp-3 correct

Output: outputs/metrics/case_study.jsonl

Usage:
  python scripts/09_case_study.py
  python scripts/09_case_study.py --n_cases 10
"""

import json
import argparse
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Loaders
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
#  Case categorization
# --------------------------------------------------------------------------- #

def categorize_exp2_cases(exp1_map: dict, exp2_records: list[dict]) -> dict[str, list]:
    """Compare Exp-1 vs Exp-2 for the same events."""
    categories = {
        "exp1_correct_exp2_wrong": [],
        "exp1_wrong_exp2_correct": [],
        "both_correct": [],
        "both_wrong": [],
    }
    for r2 in exp2_records:
        eid = r2["event_id"]
        r1 = exp1_map.get(eid)
        if not r1:
            continue
        gold = r2["gold"]
        p1 = r1["pred"]
        p2 = r2["pred"]
        c1 = (p1 == gold)
        c2 = (p2 == gold)
        if c1 and not c2:
            categories["exp1_correct_exp2_wrong"].append((r1, r2))
        elif not c1 and c2:
            categories["exp1_wrong_exp2_correct"].append((r1, r2))
        elif c1 and c2:
            categories["both_correct"].append((r1, r2))
        else:
            categories["both_wrong"].append((r1, r2))
    return categories


def categorize_exp3_cases(exp1_map: dict, exp3_records: list[dict]) -> dict[str, list]:
    """Compare Exp-1 vs Exp-3 final pred."""
    categories = {
        "exp1_correct_exp3_wrong": [],
        "exp1_wrong_exp3_correct": [],
        "both_correct": [],
        "both_wrong": [],
    }
    for r3 in exp3_records:
        eid = r3["event_id"]
        r1 = exp1_map.get(eid)
        if not r1:
            continue
        gold = r3["gold"]
        p1 = r1["pred"]
        p3 = r3["final_pred"]
        c1 = (p1 == gold)
        c3 = (p3 == gold)
        if c1 and not c3:
            categories["exp1_correct_exp3_wrong"].append((r1, r3))
        elif not c1 and c3:
            categories["exp1_wrong_exp3_correct"].append((r1, r3))
        elif c1 and c3:
            categories["both_correct"].append((r1, r3))
        else:
            categories["both_wrong"].append((r1, r3))
    return categories


# --------------------------------------------------------------------------- #
#  Case builder
# --------------------------------------------------------------------------- #

def build_case(category: str, r1: dict, r_other: dict, basepack_map: dict,
               exp2_aug_map: dict | None = None,
               exp3_full_map: dict | None = None) -> dict:
    eid = r1["event_id"]
    bp = basepack_map.get(eid, {})

    label_str = {0: "True", 1: "Fake"}

    case = {
        "event_id": eid,
        "category": category,
        "gold": r1["gold"],
        "gold_label": label_str.get(r1["gold"], str(r1["gold"])),
        "source_text": bp.get("source_text", ""),
        "selected_replies": bp.get("selected_replies", [])[:3],
        "stats": bp.get("stats", {}),
        "exp1": {
            "pred": r1["pred"],
            "pred_label": label_str.get(r1["pred"], str(r1["pred"])),
            "confidence": r1.get("confidence", None),
        },
    }

    # Exp-2 specific fields
    if "exp2" in category:
        aug_rec = exp2_aug_map.get(eid) if exp2_aug_map else None
        case["exp2"] = {
            "pred": r_other["pred"],
            "pred_label": label_str.get(r_other["pred"], str(r_other["pred"])),
            "confidence": r_other.get("confidence", None),
            "llm_aug": aug_rec.get("llm_aug") if aug_rec else None,
        }

    # Exp-3 specific fields
    if "exp3" in category:
        exp3_full = exp3_full_map.get(eid) if exp3_full_map else None
        case["exp3"] = {
            "final_pred": r_other["final_pred"],
            "final_pred_label": label_str.get(r_other["final_pred"], str(r_other["final_pred"])),
            "small_confidence": r_other.get("small_confidence", None),
            "llm_pred_label": r_other.get("llm_pred_label", "N/A"),
            "llm_reason": r_other.get("llm_parsed", {}).get("reason", "") if r_other.get("llm_parsed") else "",
        }

    return case


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Build case studies for all experiments")
    parser.add_argument("--exp1_preds", default="outputs/exp1_small_only/test_predictions.jsonl")
    parser.add_argument("--exp2_preds", default="outputs/exp2_llm_pre/test_predictions.jsonl")
    parser.add_argument("--exp3_preds", default="outputs/exp3_llm_post/test_predictions.jsonl")
    parser.add_argument("--exp2_aug", default="outputs/exp2_llm_pre/llm_aug_test.jsonl")
    parser.add_argument("--basepack", default="data/processed/basepack_test.jsonl")
    parser.add_argument("--output", default="outputs/metrics/case_study.jsonl")
    parser.add_argument("--n_cases", type=int, default=5,
                        help="Max cases per category")
    args = parser.parse_args()

    # Load all data
    exp1_records = load_jsonl(Path(args.exp1_preds))
    exp2_records = load_jsonl(Path(args.exp2_preds))
    exp3_records = load_jsonl(Path(args.exp3_preds))
    basepacks = {e["event_id"]: e for e in load_jsonl(Path(args.basepack))}
    exp2_augs = {r["event_id"]: r for r in load_jsonl(Path(args.exp2_aug))} if Path(args.exp2_aug).exists() else {}
    exp3_full = {r["event_id"]: r for r in exp3_records}

    exp1_map = {r["event_id"]: r for r in exp1_records}

    all_cases = []

    # Exp-1 vs Exp-2
    if exp2_records:
        cats2 = categorize_exp2_cases(exp1_map, exp2_records)
        print(f"\nExp-1 vs Exp-2:")
        for cat, pairs in cats2.items():
            print(f"  {cat}: {len(pairs)}")

        for cat_key in ["exp1_wrong_exp2_correct", "exp1_correct_exp2_wrong"]:
            for r1, r2 in cats2[cat_key][:args.n_cases]:
                case = build_case(
                    category=cat_key,
                    r1=r1, r_other=r2,
                    basepack_map=basepacks,
                    exp2_aug_map=exp2_augs,
                )
                all_cases.append(case)

    # Exp-1 vs Exp-3
    if exp3_records:
        cats3 = categorize_exp3_cases(exp1_map, exp3_records)
        print(f"\nExp-1 vs Exp-3:")
        for cat, pairs in cats3.items():
            print(f"  {cat}: {len(pairs)}")

        for cat_key in ["exp1_wrong_exp3_correct", "exp1_correct_exp3_wrong"]:
            for r1, r3 in cats3[cat_key][:args.n_cases]:
                case = build_case(
                    category=cat_key,
                    r1=r1, r_other=r3,
                    basepack_map=basepacks,
                    exp3_full_map=exp3_full,
                )
                all_cases.append(case)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nCase study saved: {len(all_cases)} cases -> {output_path}")

    # Print summary
    cat_counts = {}
    for c in all_cases:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    for cat, cnt in cat_counts.items():
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
