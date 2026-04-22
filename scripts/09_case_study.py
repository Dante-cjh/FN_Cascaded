"""
09_case_study.py
Build case studies comparing all experiments.

For each of the four interesting case categories, selects up to n_cases examples:
  A) Exp-1 correct, Exp-2   wrong
  B) Exp-1 wrong,   Exp-2   correct
  C) Exp-1 correct, Exp-3   wrong
  D) Exp-1 wrong,   Exp-3   correct
  E) Exp-1 correct, Exp-3b  wrong   (Thinking version)
  F) Exp-1 wrong,   Exp-3b  correct (Thinking version)
  G) Exp-3 correct, Exp-3b  wrong   (Thinking 变差)
  H) Exp-3 wrong,   Exp-3b  correct (Thinking 改善)

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


def categorize_exp3_cases(exp1_map: dict, exp3_records: list[dict],
                          prefix: str = "exp3") -> dict[str, list]:
    """Compare Exp-1 vs Exp-3 (or Exp-3b) final pred.

    prefix: 'exp3' for standard post, 'exp3b' for thinking variant.
    """
    categories = {
        f"exp1_correct_{prefix}_wrong": [],
        f"exp1_wrong_{prefix}_correct": [],
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
            categories[f"exp1_correct_{prefix}_wrong"].append((r1, r3))
        elif not c1 and c3:
            categories[f"exp1_wrong_{prefix}_correct"].append((r1, r3))
        elif c1 and c3:
            categories["both_correct"].append((r1, r3))
        else:
            categories["both_wrong"].append((r1, r3))
    return categories


def categorize_exp3_vs_exp3b(exp3_map: dict, exp3b_records: list[dict]) -> dict[str, list]:
    """Compare Exp-3 vs Exp-3b，分析 thinking 带来的改变。"""
    categories = {
        "exp3_correct_exp3b_wrong":  [],   # thinking 引入错误
        "exp3_wrong_exp3b_correct":  [],   # thinking 纠正错误
        "both_correct":              [],
        "both_wrong":                [],
    }
    for r3b in exp3b_records:
        eid = r3b["event_id"]
        r3 = exp3_map.get(eid)
        if not r3:
            continue
        gold = r3b["gold"]
        p3  = r3["final_pred"]
        p3b = r3b["final_pred"]
        c3  = (p3  == gold)
        c3b = (p3b == gold)
        if c3 and not c3b:
            categories["exp3_correct_exp3b_wrong"].append((r3, r3b))
        elif not c3 and c3b:
            categories["exp3_wrong_exp3b_correct"].append((r3, r3b))
        elif c3 and c3b:
            categories["both_correct"].append((r3, r3b))
        else:
            categories["both_wrong"].append((r3, r3b))
    return categories


# --------------------------------------------------------------------------- #
#  Case builder
# --------------------------------------------------------------------------- #

def _exp3_fields(r: dict) -> dict:
    """提取 Exp-3 / Exp-3b 预测记录中的关键字段。"""
    label_str = {0: "True", 1: "Fake"}
    return {
        "final_pred":       r["final_pred"],
        "final_pred_label": label_str.get(r["final_pred"], str(r["final_pred"])),
        "small_confidence": r.get("small_confidence", None),
        "llm_pred_label":   r.get("llm_pred_label", "N/A"),
        "llm_reason": (
            r.get("llm_parsed", {}).get("reason", "")
            if r.get("llm_parsed") else ""
        ),
        "tokens_used": r.get("tokens_used", 0),
    }


def build_case(category: str, r1: dict, r_other: dict, basepack_map: dict,
               exp2_aug_map: dict | None = None,
               exp3_full_map: dict | None = None,
               exp3b_full_map: dict | None = None) -> dict:
    """构建单个案例字典。

    category 可能的前缀：
      exp2       → Exp-1 vs Exp-2 对比
      exp3b      → Exp-1 vs Exp-3b 对比（thinking）
      exp3_vs_   → Exp-3 vs Exp-3b 直接对比（thinking effect）
      exp3（其余）→ Exp-1 vs Exp-3 对比
    """
    label_str = {0: "True", 1: "Fake"}
    eid = r1["event_id"]
    bp  = basepack_map.get(eid, {})

    # exp3_vs_exp3b 类别的 r1 实际上是 exp3 记录，gold 从其中取
    gold_val = r1.get("gold", r_other.get("gold", -1))

    case = {
        "event_id":         eid,
        "category":         category,
        "gold":             gold_val,
        "gold_label":       label_str.get(gold_val, str(gold_val)),
        "source_text":      bp.get("source_text", ""),
        "selected_replies": bp.get("selected_replies", [])[:3],
        "stats":            bp.get("stats", {}),
    }

    # exp3_vs_exp3b：r1 = exp3 记录，r_other = exp3b 记录
    if category.startswith("exp3_vs_") or category.startswith("exp3_correct_exp3b") \
            or category.startswith("exp3_wrong_exp3b"):
        case["exp3"]  = _exp3_fields(r1)
        case["exp3b"] = _exp3_fields(r_other)
        return case

    # 其余情况都需要 exp1 信息
    case["exp1"] = {
        "pred":       r1["pred"],
        "pred_label": label_str.get(r1["pred"], str(r1["pred"])),
        "confidence": r1.get("confidence", None),
    }

    if "exp2" in category:
        aug_rec = exp2_aug_map.get(eid) if exp2_aug_map else None
        case["exp2"] = {
            "pred":       r_other["pred"],
            "pred_label": label_str.get(r_other["pred"], str(r_other["pred"])),
            "confidence": r_other.get("confidence", None),
            "llm_aug":    aug_rec.get("llm_aug") if aug_rec else None,
        }
    elif "exp3b" in category:
        case["exp3b"] = _exp3_fields(r_other)
    elif "exp3" in category:
        case["exp3"] = _exp3_fields(r_other)

    return case


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Build case studies for all experiments")
    parser.add_argument("--exp1_preds",  default="outputs/exp1_small_only/test_predictions.jsonl")
    parser.add_argument("--exp2_preds",  default="outputs/exp2_llm_pre/test_predictions.jsonl")
    parser.add_argument("--exp3_preds",  default="outputs/exp3_llm_post/test_predictions.jsonl")
    parser.add_argument("--exp3b_preds", default="outputs/exp3_llm_post/test_predictions_thinking.jsonl",
                        help="Exp-3b: LLM-Post with enable_thinking=True")
    parser.add_argument("--exp2_aug",    default="outputs/exp2_llm_pre/llm_aug_test.jsonl")
    parser.add_argument("--basepack",    default="data/processed/basepack_test.jsonl")
    parser.add_argument("--output",      default="outputs/metrics/case_study.jsonl")
    parser.add_argument("--n_cases", type=int, default=5,
                        help="Max cases per category")
    args = parser.parse_args()

    # 加载所有数据
    exp1_records  = load_jsonl(Path(args.exp1_preds))
    exp2_records  = load_jsonl(Path(args.exp2_preds))
    exp3_records  = load_jsonl(Path(args.exp3_preds))
    exp3b_records = load_jsonl(Path(args.exp3b_preds))
    basepacks = {e["event_id"]: e for e in load_jsonl(Path(args.basepack))}
    exp2_augs = (
        {r["event_id"]: r for r in load_jsonl(Path(args.exp2_aug))}
        if Path(args.exp2_aug).exists() else {}
    )
    exp3_full  = {r["event_id"]: r for r in exp3_records}
    exp3b_full = {r["event_id"]: r for r in exp3b_records}

    exp1_map = {r["event_id"]: r for r in exp1_records}
    exp3_map = {r["event_id"]: r for r in exp3_records}

    all_cases = []

    # ── Exp-1 vs Exp-2 ────────────────────────────────────────────────────
    if exp2_records:
        cats2 = categorize_exp2_cases(exp1_map, exp2_records)
        print(f"\nExp-1 vs Exp-2:")
        for cat, pairs in cats2.items():
            print(f"  {cat}: {len(pairs)}")

        for cat_key in ["exp1_wrong_exp2_correct", "exp1_correct_exp2_wrong"]:
            for r1, r2 in cats2[cat_key][:args.n_cases]:
                all_cases.append(build_case(
                    category=cat_key,
                    r1=r1, r_other=r2,
                    basepack_map=basepacks,
                    exp2_aug_map=exp2_augs,
                ))

    # ── Exp-1 vs Exp-3 ────────────────────────────────────────────────────
    if exp3_records:
        cats3 = categorize_exp3_cases(exp1_map, exp3_records, prefix="exp3")
        print(f"\nExp-1 vs Exp-3:")
        for cat, pairs in cats3.items():
            print(f"  {cat}: {len(pairs)}")

        for cat_key in ["exp1_wrong_exp3_correct", "exp1_correct_exp3_wrong"]:
            for r1, r3 in cats3[cat_key][:args.n_cases]:
                all_cases.append(build_case(
                    category=cat_key,
                    r1=r1, r_other=r3,
                    basepack_map=basepacks,
                    exp3_full_map=exp3_full,
                ))

    # ── Exp-1 vs Exp-3b (Thinking) ────────────────────────────────────────
    if exp3b_records:
        cats3b = categorize_exp3_cases(exp1_map, exp3b_records, prefix="exp3b")
        print(f"\nExp-1 vs Exp-3b (Thinking):")
        for cat, pairs in cats3b.items():
            print(f"  {cat}: {len(pairs)}")

        for cat_key in ["exp1_wrong_exp3b_correct", "exp1_correct_exp3b_wrong"]:
            for r1, r3b in cats3b[cat_key][:args.n_cases]:
                all_cases.append(build_case(
                    category=cat_key,
                    r1=r1, r_other=r3b,
                    basepack_map=basepacks,
                    exp3b_full_map=exp3b_full,
                ))

    # ── Exp-3 vs Exp-3b (Thinking effect) ────────────────────────────────
    if exp3_records and exp3b_records:
        cats_diff = categorize_exp3_vs_exp3b(exp3_map, exp3b_records)
        print(f"\nExp-3 vs Exp-3b (Thinking effect):")
        for cat, pairs in cats_diff.items():
            print(f"  {cat}: {len(pairs)}")

        for cat_key in ["exp3_wrong_exp3b_correct", "exp3_correct_exp3b_wrong"]:
            for r3, r3b in cats_diff[cat_key][:args.n_cases]:
                all_cases.append(build_case(
                    category=cat_key,
                    r1=r3, r_other=r3b,
                    basepack_map=basepacks,
                    exp3_full_map=exp3_full,
                    exp3b_full_map=exp3b_full,
                ))

    # ── 写入输出 ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nCase study saved: {len(all_cases)} cases -> {output_path}")

    # 打印汇总
    cat_counts: dict[str, int] = {}
    for c in all_cases:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    for cat, cnt in cat_counts.items():
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
