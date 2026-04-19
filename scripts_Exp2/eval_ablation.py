"""
scripts_Exp2/eval_ablation.py
Evaluate all 7 Exp-2 ablation conditions and print a comparison table.

Reads:  outputs/exp2_ablation/abl{1..7}_*/test_predictions.jsonl
Writes: outputs/exp2_ablation/metrics/ablation_results.{txt,md,json}

Each test_predictions.jsonl must contain lines with at least:
  {"gold": int, "pred": int, ...}

Run from project root:
  python scripts_Exp2/eval_ablation.py
  python scripts_Exp2/eval_ablation.py --pred_dir outputs/exp2_ablation
"""

import json
import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------------------------------- #
#  Condition registry — order matches ablation numbering
# --------------------------------------------------------------------------- #

CONDITIONS = [
    {
        "id":     "abl1",
        "name":   "abl1_basepack_only",
        "label":  "BasePack only",
        "fields": "—",
    },
    {
        "id":     "abl2",
        "name":   "abl2_claim_only",
        "label":  "BasePack + claim_summary",
        "fields": "claim_summary",
    },
    {
        "id":     "abl3",
        "name":   "abl3_claim_signals",
        "label":  "BasePack + claim + supporting/refuting",
        "fields": "claim_summary, supporting_signals, refuting_signals",
    },
    {
        "id":     "abl4",
        "name":   "abl4_signals_only",
        "label":  "BasePack + supporting/refuting only",
        "fields": "supporting_signals, refuting_signals",
    },
    {
        "id":     "abl5",
        "name":   "abl5_analysis_only",
        "label":  "BasePack + conflict_summary + risk_note",
        "fields": "conflict_summary, risk_note",
    },
    {
        "id":     "abl6",
        "name":   "abl6_claim_analysis",
        "label":  "BasePack + claim + conflict_summary + risk_note",
        "fields": "claim_summary, conflict_summary, risk_note",
    },
    {
        "id":     "abl7",
        "name":   "abl7_full_llm_aug",
        "label":  "BasePack + full LLM aug",
        "fields": "claim_summary, supporting_signals, refuting_signals, conflict_summary, risk_note",
    },
]


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(pred_path: Path) -> dict | None:
    if not pred_path.exists():
        return None

    golds, preds = [], []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                golds.append(rec["gold"])
                preds.append(rec["pred"])

    if not golds:
        return None

    acc      = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro",    zero_division=0)
    f1_true  = f1_score(golds, preds, pos_label=0, average="binary", zero_division=0)
    f1_fake  = f1_score(golds, preds, pos_label=1, average="binary", zero_division=0)

    return {
        "n":        len(golds),
        "acc":      acc,
        "macro_f1": macro_f1,
        "f1_true":  f1_true,
        "f1_fake":  f1_fake,
    }


# --------------------------------------------------------------------------- #
#  Formatting helpers
# --------------------------------------------------------------------------- #

def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  —  "
    return f"{v * 100:.1f}" if pct else f"{v:.4f}"


def build_plain_table(results: list[dict]) -> str:
    col_w = [4, 42, 8, 9, 8, 8, 6]
    header = (
        f"{'ID':<{col_w[0]}}  "
        f"{'Condition':<{col_w[1]}}  "
        f"{'Acc (%)':>{col_w[2]}}  "
        f"{'MacroF1':>{col_w[3]}}  "
        f"{'F1-True':>{col_w[4]}}  "
        f"{'F1-Fake':>{col_w[5]}}  "
        f"{'N':>{col_w[6]}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    best_macro = max(
        (r["metrics"]["macro_f1"] for r in results if r["metrics"]),
        default=None,
    )

    for r in results:
        m = r["metrics"]
        marker = " *" if (m and best_macro is not None and abs(m["macro_f1"] - best_macro) < 1e-6) else "  "
        acc_s      = fmt(m["acc"])      if m else "  —  "
        macro_s    = fmt(m["macro_f1"]) if m else "  —  "
        f1_true_s  = fmt(m["f1_true"])  if m else "  —  "
        f1_fake_s  = fmt(m["f1_fake"])  if m else "  —  "
        n_s        = str(m["n"])        if m else "—"

        line = (
            f"{r['id']:<{col_w[0]}}  "
            f"{r['label']:<{col_w[1]}}  "
            f"{acc_s:>{col_w[2]}}  "
            f"{macro_s:>{col_w[3]}}  "
            f"{f1_true_s:>{col_w[4]}}  "
            f"{f1_fake_s:>{col_w[5]}}  "
            f"{n_s:>{col_w[6]}}"
            f"{marker}"
        )
        lines.append(line)

    lines.append(sep)
    lines.append("* = best macro-F1")
    return "\n".join(lines)


def build_markdown_table(results: list[dict]) -> str:
    best_macro = max(
        (r["metrics"]["macro_f1"] for r in results if r["metrics"]),
        default=None,
    )
    lines = [
        "| ID | Condition | Acc (%) | Macro-F1 | F1-True | F1-Fake | N |",
        "|:---|:----------|--------:|---------:|--------:|--------:|--:|",
    ]
    for r in results:
        m = r["metrics"]
        bold = (m and best_macro is not None and abs(m["macro_f1"] - best_macro) < 1e-6)
        acc_s     = fmt(m["acc"])      if m else "—"
        macro_s   = fmt(m["macro_f1"]) if m else "—"
        f1_true_s = fmt(m["f1_true"])  if m else "—"
        f1_fake_s = fmt(m["f1_fake"])  if m else "—"
        n_s       = str(m["n"])        if m else "—"

        if bold:
            macro_s = f"**{macro_s}**"

        lines.append(
            f"| {r['id']} | {r['label']} "
            f"| {acc_s} | {macro_s} | {f1_true_s} | {f1_fake_s} | {n_s} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Exp-2 ablation conditions"
    )
    parser.add_argument(
        "--pred_dir", default="outputs/exp2_ablation",
        help="Root directory containing abl*/ subdirs with test_predictions.jsonl",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write metric files (default: {pred_dir}/metrics)",
    )
    args = parser.parse_args()

    pred_dir   = Path(args.pred_dir)
    output_dir = Path(args.output_dir) if args.output_dir else pred_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cond in CONDITIONS:
        pred_path = pred_dir / cond["name"] / "test_predictions.jsonl"
        metrics   = compute_metrics(pred_path)
        results.append({
            "id":      cond["id"],
            "name":    cond["name"],
            "label":   cond["label"],
            "fields":  cond["fields"],
            "metrics": metrics,
            "missing": metrics is None,
        })
        status = "OK" if metrics else "MISSING"
        print(f"  [{status}] {cond['name']}")

    plain_table = build_plain_table(results)
    md_table    = build_markdown_table(results)

    print()
    print(plain_table)

    # ---- plain text ----
    txt_path = output_dir / "ablation_results.txt"
    with open(txt_path, "w") as f:
        f.write("Exp-2 Ablation Study — Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(plain_table + "\n\n")
        f.write("Field legend\n")
        f.write("-" * 40 + "\n")
        for cond in CONDITIONS:
            f.write(f"  {cond['id']}: {cond['fields']}\n")
    print(f"\nPlain text  -> {txt_path}")

    # ---- markdown ----
    md_path = output_dir / "ablation_results.md"
    with open(md_path, "w") as f:
        f.write("## Exp-2 Ablation Study — Results\n\n")
        f.write(md_table + "\n\n")
        f.write("### Field legend\n\n")
        for cond in CONDITIONS:
            f.write(f"- **{cond['id']}**: {cond['fields']}\n")
    print(f"Markdown    -> {md_path}")

    # ---- JSON ----
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON        -> {json_path}")

    # ---- delta vs abl1 ----
    base_metrics = results[0]["metrics"]
    if base_metrics:
        print("\nDelta vs abl1 (BasePack only)  [Macro-F1 pp]")
        print("-" * 50)
        for r in results[1:]:
            if r["metrics"]:
                delta = (r["metrics"]["macro_f1"] - base_metrics["macro_f1"]) * 100
                sign  = "+" if delta >= 0 else ""
                print(f"  {r['id']}: {sign}{delta:.1f} pp")


if __name__ == "__main__":
    main()
