"""
scripts_Exp2_dir2/eval_ablation_dir2.py
Evaluate all 8 leave-one-out (w/o) ablation conditions for Dir2 (Narrative/Manipulation).

Shows the performance DROP when each field is removed.

Reads:  outputs/exp2_ablation_dir2/{dir2_full, dir2_wo_*}/test_predictions.jsonl
Writes: outputs/exp2_ablation_dir2/metrics/ablation_dir2_results.{txt,md,json}

Run from project root:
  python scripts_Exp2_dir2/eval_ablation_dir2.py
"""

import json
import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------------------------------- #
#  Condition registry
# --------------------------------------------------------------------------- #

CONDITIONS = [
    {"id": "full",                        "name": "dir2_full",                       "label": "Full (all 7 fields)",              "removed": "—"},
    {"id": "wo_narrative_frame",          "name": "dir2_wo_narrative_frame",         "label": "w/o narrative_frame",              "removed": "narrative_frame"},
    {"id": "wo_persuasion_cues",          "name": "dir2_wo_persuasion_cues",         "label": "w/o persuasion_cues",              "removed": "persuasion_cues"},
    {"id": "wo_engagement_pattern",       "name": "dir2_wo_engagement_pattern",      "label": "w/o engagement_pattern",           "removed": "engagement_pattern"},
    {"id": "wo_coordination_signals",     "name": "dir2_wo_coordination_signals",    "label": "w/o coordination_signals",         "removed": "coordination_signals"},
    {"id": "wo_evidence_visibility",      "name": "dir2_wo_evidence_visibility",     "label": "w/o evidence_visibility",          "removed": "evidence_visibility"},
    {"id": "wo_attention_triggers",       "name": "dir2_wo_attention_triggers",      "label": "w/o attention_triggers",           "removed": "attention_triggers"},
    {"id": "wo_manipulation_risk_profile","name": "dir2_wo_manipulation_risk_profile","label": "w/o manipulation_risk_profile",   "removed": "manipulation_risk_profile"},
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

    return {
        "n":        len(golds),
        "acc":      accuracy_score(golds, preds),
        "macro_f1": f1_score(golds, preds, average="macro",    zero_division=0),
        "f1_true":  f1_score(golds, preds, pos_label=0, average="binary", zero_division=0),
        "f1_fake":  f1_score(golds, preds, pos_label=1, average="binary", zero_division=0),
    }


def fmt(v: float | None) -> str:
    return f"{v * 100:.2f}" if v is not None else "  —  "


def fmt_delta(v: float | None) -> str:
    if v is None:
        return "   —  "
    return f"{v * 100:+.2f}"


# --------------------------------------------------------------------------- #
#  Table builders
# --------------------------------------------------------------------------- #

def build_plain_table(results: list[dict], full_metrics: dict | None) -> str:
    hdr = (
        f"{'Condition':<32}  "
        f"{'Acc':>6}  {'MacroF1':>8}  {'F1-True':>8}  {'F1-Fake':>8}  "
        f"{'ΔAcc':>7}  {'ΔMacroF1':>9}  {'ΔF1-Fake':>9}"
    )
    sep = "-" * len(hdr)
    lines = [sep, hdr, sep]

    for r in results:
        m = r["metrics"]
        is_full = (r["id"] == "full")

        acc_s   = fmt(m["acc"])      if m else "  —  "
        macro_s = fmt(m["macro_f1"]) if m else "  —  "
        f1t_s   = fmt(m["f1_true"])  if m else "  —  "
        f1f_s   = fmt(m["f1_fake"])  if m else "  —  "

        if is_full or not m or not full_metrics:
            d_acc = d_macro = d_fake = "    —  "
        else:
            d_acc   = fmt_delta(m["acc"]      - full_metrics["acc"])
            d_macro = fmt_delta(m["macro_f1"] - full_metrics["macro_f1"])
            d_fake  = fmt_delta(m["f1_fake"]  - full_metrics["f1_fake"])

        line = (
            f"{r['label']:<32}  "
            f"{acc_s:>6}  {macro_s:>8}  {f1t_s:>8}  {f1f_s:>8}  "
            f"{d_acc:>7}  {d_macro:>9}  {d_fake:>9}"
        )
        if is_full:
            lines.append(line)
            lines.append(sep)
        else:
            lines.append(line)

    lines.append(sep)
    lines.append("Δ = (w/o condition) − Full.  Negative = that field HELPS.")
    return "\n".join(lines)


def build_markdown_table(results: list[dict], full_metrics: dict | None) -> str:
    lines = [
        "| Condition | Acc (%) | Macro-F1 | F1-True | F1-Fake | ΔAcc | ΔMacro-F1 | ΔF1-Fake |",
        "|:----------|--------:|---------:|--------:|--------:|-----:|----------:|---------:|",
    ]

    for r in results:
        m = r["metrics"]
        is_full = (r["id"] == "full")
        acc_s   = fmt(m["acc"])      if m else "—"
        macro_s = fmt(m["macro_f1"]) if m else "—"
        f1t_s   = fmt(m["f1_true"])  if m else "—"
        f1f_s   = fmt(m["f1_fake"])  if m else "—"

        if is_full:
            macro_s = f"**{macro_s}**"
            d_acc = d_macro = d_fake = "—"
        elif m and full_metrics:
            d_acc   = fmt_delta(m["acc"]      - full_metrics["acc"])
            d_macro = fmt_delta(m["macro_f1"] - full_metrics["macro_f1"])
            d_fake  = fmt_delta(m["f1_fake"]  - full_metrics["f1_fake"])
        else:
            d_acc = d_macro = d_fake = "—"

        lines.append(
            f"| {r['label']} | {acc_s} | {macro_s} | {f1t_s} | {f1f_s}"
            f" | {d_acc} | {d_macro} | {d_fake} |"
        )

    return "\n".join(lines)


def build_ranking(results: list[dict], full_metrics: dict | None) -> str:
    if not full_metrics:
        return "(full model results not available — cannot rank)"

    drops = []
    for r in results:
        if r["id"] == "full" or not r["metrics"]:
            continue
        drop_macro = full_metrics["macro_f1"] - r["metrics"]["macro_f1"]
        drop_fake  = full_metrics["f1_fake"]  - r["metrics"]["f1_fake"]
        drops.append((r["removed"], drop_macro, drop_fake))

    drops.sort(key=lambda x: x[1], reverse=True)

    lines = ["Field importance ranking  (by Macro-F1 drop when removed)",
             "-" * 62,
             f"  {'Field':<29}  {'ΔMacroF1 (pp)':>14}  {'ΔF1-Fake (pp)':>14}"]
    for field, dm, df in drops:
        lines.append(f"  {field:<29}  {dm * 100:>+14.2f}  {df * 100:>+14.2f}")
    lines.append("")
    lines.append("  Positive = performance DROP when field is removed")
    lines.append("  (more positive = field contributes MORE to the model)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Exp-2 Dir2 leave-one-out ablation conditions"
    )
    parser.add_argument(
        "--pred_dir", default="outputs/exp2_ablation_dir2",
        help="Root dir containing dir2_*/ subdirs with test_predictions.jsonl",
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
        results.append({**cond, "metrics": metrics, "missing": metrics is None})
        status = "OK" if metrics else "MISSING"
        print(f"  [{status}] {cond['name']}")

    full_metrics = next(
        (r["metrics"] for r in results if r["id"] == "full"), None
    )

    plain_table = build_plain_table(results, full_metrics)
    md_table    = build_markdown_table(results, full_metrics)
    ranking     = build_ranking(results, full_metrics)

    print()
    print(plain_table)
    print()
    print(ranking)

    txt_path = output_dir / "ablation_dir2_results.txt"
    with open(txt_path, "w") as f:
        f.write("Exp-2 Dir2 (Narrative/Manipulation) Leave-One-Out (W/O) Ablation — Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(plain_table + "\n\n")
        f.write(ranking + "\n")
    print(f"\nPlain text -> {txt_path}")

    md_path = output_dir / "ablation_dir2_results.md"
    with open(md_path, "w") as f:
        f.write("## Exp-2 Dir2 (Narrative/Manipulation) Leave-One-Out (W/O) Ablation — Results\n\n")
        f.write(md_table + "\n\n")
        f.write("> Δ = (w/o condition) − Full.  "
                "Negative Δ means the field HELPS the model.\n\n")
        f.write("### Field Importance Ranking\n\n")
        f.write("```\n" + ranking + "\n```\n")
    print(f"Markdown   -> {md_path}")

    json_path = output_dir / "ablation_dir2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON       -> {json_path}")


if __name__ == "__main__":
    main()
