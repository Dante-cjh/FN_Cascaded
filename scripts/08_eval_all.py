"""
08_eval_all.py
Unified evaluation for all controlled variable experiments.

Exp-1 : Small-Only              (outputs/exp1_small_only/test_predictions.jsonl)
Exp-2 : LLM-Pre + Small         (outputs/exp2_llm_pre/test_predictions.jsonl)
Exp-3 : Small + LLM-Post        (outputs/exp3_llm_post/test_predictions.jsonl)
Exp-3b: Small + LLM-Post(Think) (outputs/exp3_llm_post/test_predictions_thinking.jsonl)

Outputs (in outputs/metrics/):
  main_results.txt / .md      - Table 1: accuracy / macro-F1 / F1-Fake / F1-True
  llm_stats.txt    / .md      - Table 2: LLM token & parse success stats
  correction_analysis.txt/.md - Table 3: flip / correction / damage rates
  all_results.json            - machine-readable full results

Usage:
  python scripts/08_eval_all.py
  python scripts/08_eval_all.py --metrics_dir outputs/metrics
"""

import json
import argparse
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


LABEL_NAMES = ["True", "Fake"]   # label 0 = True, label 1 = Fake


# --------------------------------------------------------------------------- #
#  Loading helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(golds: list[int], preds: list[int]) -> dict:
    acc       = accuracy_score(golds, preds)
    macro_f1  = f1_score(golds, preds, average="macro",  zero_division=0)
    f1_true   = f1_score(golds, preds, pos_label=0, average="binary", zero_division=0)
    f1_fake   = f1_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    prec_fake = precision_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    prec_true = precision_score(golds, preds, pos_label=0, average="binary", zero_division=0)
    rec_fake  = recall_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    rec_true  = recall_score(golds, preds, pos_label=0, average="binary", zero_division=0)
    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "f1_fake":   f1_fake,
        "f1_true":   f1_true,
        "prec_fake": prec_fake,
        "prec_true": prec_true,
        "rec_fake":  rec_fake,
        "rec_true":  rec_true,
    }


# --------------------------------------------------------------------------- #
#  Per-experiment evaluation
# --------------------------------------------------------------------------- #

def eval_exp(name: str, pred_path: Path,
             pred_field: str = "pred", gold_field: str = "gold") -> dict | None:
    records = load_jsonl(pred_path)
    if not records:
        print(f"[WARN] {name} predictions not found: {pred_path}")
        return None
    golds = [r[gold_field] for r in records]
    preds = [r[pred_field] for r in records]
    metrics = compute_metrics(golds, preds)
    print(f"\n=== {name} ===")
    print(classification_report(golds, preds, target_names=LABEL_NAMES, zero_division=0))
    return {"exp": name, "metrics": metrics, "n": len(records)}


# --------------------------------------------------------------------------- #
#  Markdown helpers
# --------------------------------------------------------------------------- #

def md_row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def md_sep(*aligns) -> str:
    """aligns: 'l', 'c', 'r'"""
    parts = []
    for a in aligns:
        if a == "r":
            parts.append("---:")
        elif a == "c":
            parts.append(":---:")
        else:
            parts.append("---")
    return "| " + " | ".join(parts) + " |"


# --------------------------------------------------------------------------- #
#  Table 1: Main results
# --------------------------------------------------------------------------- #

def print_main_table(r1, r2, r3, r3b, out_dir: Path):
    results = [r1, r2, r3, r3b]
    meta = [
        ("Exp-1  Small-Only",         "Yes", "None"),
        ("Exp-2  LLM-Pre+Small",      "Yes", "Pre"),
        ("Exp-3  Small+LLM-Post",     "Yes", "Post"),
        ("Exp-3b Small+LLM-Post(Think)", "Yes", "Post+Think"),
    ]

    # ── plain text ─────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 96)
    lines.append("Table 1: Main Results (Fake/True Binary, PHEME)")
    lines.append("=" * 96)
    hdr = (f"{'Method':<30} {'Small Fixed':>11} {'LLM Pos':>11} "
           f"{'Accuracy':>9} {'Macro-F1':>9} {'F1-Fake':>8} {'F1-True':>8}")
    lines.append(hdr)
    lines.append("-" * 96)
    for r, (name, fixed, pos) in zip(results, meta):
        if r is None:
            lines.append(f"{name:<30} {fixed:>11} {pos:>11} "
                         f"{'N/A':>9} {'N/A':>9} {'N/A':>8} {'N/A':>8}")
        else:
            m = r["metrics"]
            lines.append(f"{name:<30} {fixed:>11} {pos:>11} "
                         f"{m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
                         f"{m['f1_fake']:>8.4f} {m['f1_true']:>8.4f}")
    lines.append("=" * 96)

    # 增量对比行（均相对于 Exp-1 基线）
    def gain_row(label, rx, ry):
        if rx is None or ry is None:
            return
        lines.append(f"\n{label}")
        for k in ["accuracy", "macro_f1", "f1_fake", "f1_true"]:
            g = ry["metrics"][k] - rx["metrics"][k]
            lines.append(f"  {k:12}: {g:+.4f}")

    gain_row("Gain: Exp-2  − Exp-1", r1, r2)
    gain_row("Gain: Exp-3  − Exp-1", r1, r3)
    gain_row("Gain: Exp-3b − Exp-1", r1, r3b)
    gain_row("Gain: Exp-3b − Exp-3 (Thinking effect)", r3, r3b)

    txt = "\n".join(lines)
    print("\n" + txt)
    _write(out_dir / "main_results.txt", txt)

    # ── markdown ────────────────────────────────────────────────────────────
    md = []
    md.append("## Table 1: Main Results\n")
    md.append(md_row("Method", "Small Fixed?", "LLM Position",
                     "Accuracy", "Macro-F1", "F1-Fake", "F1-True"))
    md.append(md_sep("l", "c", "c", "r", "r", "r", "r"))
    for r, (name, fixed, pos) in zip(results, meta):
        if r is None:
            md.append(md_row(name, fixed, pos, "N/A", "N/A", "N/A", "N/A"))
        else:
            m = r["metrics"]
            md.append(md_row(
                name, fixed, pos,
                f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}",
                f"{m['f1_fake']:.4f}",  f"{m['f1_true']:.4f}",
            ))
    md.append("")

    def md_gain(label, rx, ry):
        if rx is None or ry is None:
            return
        md.append(f"**{label}**\n")
        md.append(md_row("Metric", "Δ"))
        md.append(md_sep("l", "r"))
        for k in ["accuracy", "macro_f1", "f1_fake", "f1_true"]:
            g = ry["metrics"][k] - rx["metrics"][k]
            md.append(md_row(k, f"{g:+.4f}"))
        md.append("")

    md_gain("Gain: Exp-2  − Exp-1", r1, r2)
    md_gain("Gain: Exp-3  − Exp-1", r1, r3)
    md_gain("Gain: Exp-3b − Exp-1", r1, r3b)
    md_gain("Gain: Exp-3b − Exp-3 (Thinking effect)", r3, r3b)

    _write(out_dir / "main_results.md", "\n".join(md))
    print(f"Saved -> {out_dir / 'main_results.txt'}  and  main_results.md")


# --------------------------------------------------------------------------- #
#  Table 2: LLM stats
# --------------------------------------------------------------------------- #

def _llm_stats_pre(llm_aug_dir: Path, splits=("train", "val", "test")) -> dict:
    total_tokens = total_events = parse_ok = 0
    for split in splits:
        for r in load_jsonl(llm_aug_dir / f"llm_aug_{split}.jsonl"):
            total_events += 1
            total_tokens += r.get("tokens_used", 0)
            if r.get("parse_success"):
                parse_ok += 1
    return {
        "total_events": total_events,
        "total_tokens": total_tokens,
        "avg_tokens":   total_tokens / max(total_events, 1),
        "parse_rate":   parse_ok / max(total_events, 1),
    }


def _llm_stats_post(pred_path: Path) -> dict:
    records = load_jsonl(pred_path)
    total   = len(records)
    tokens  = sum(r.get("tokens_used", 0) for r in records)
    ok      = sum(1 for r in records if r.get("llm_pred", -1) >= 0)
    return {
        "total_events": total,
        "total_tokens": tokens,
        "avg_tokens":   tokens / max(total, 1),
        "parse_rate":   ok / max(total, 1),
    }


def print_llm_stats_table(stats2, stats3, stats3b, out_dir: Path):
    rows = [
        ("Exp-2  LLM-Pre+Small",         stats2),
        ("Exp-3  Small+LLM-Post",        stats3),
        ("Exp-3b Small+LLM-Post(Think)", stats3b),
    ]

    # plain text
    lines = ["=" * 72, "Table 2: LLM Cost & Reliability", "=" * 72,
             f"{'Method':<30} {'Avg Tokens':>12} {'Total Tokens':>14} {'Parse Succ':>12}",
             "-" * 72]
    for name, st in rows:
        if st:
            lines.append(f"{name:<30} {st['avg_tokens']:>12.0f} "
                         f"{st['total_tokens']:>14} {st['parse_rate']:>12.1%}")
        else:
            lines.append(f"{name:<30} {'N/A':>12} {'N/A':>14} {'N/A':>12}")
    lines.append("=" * 72)
    txt = "\n".join(lines)
    print("\n" + txt)
    _write(out_dir / "llm_stats.txt", txt)

    # markdown
    md = ["## Table 2: LLM Cost & Reliability\n",
          md_row("Method", "Avg Tokens", "Total Tokens", "Parse Success"),
          md_sep("l", "r", "r", "r")]
    for name, st in rows:
        if st:
            md.append(md_row(name, f"{st['avg_tokens']:.0f}",
                             st["total_tokens"], f"{st['parse_rate']:.1%}"))
        else:
            md.append(md_row(name, "N/A", "N/A", "N/A"))
    _write(out_dir / "llm_stats.md", "\n".join(md))
    print(f"Saved -> {out_dir / 'llm_stats.txt'}  and  llm_stats.md")


# --------------------------------------------------------------------------- #
#  Table 3: Correction analysis
# --------------------------------------------------------------------------- #

def compute_correction_analysis(exp1_path: Path, exp3_path: Path) -> dict | None:
    exp1 = {r["event_id"]: r for r in load_jsonl(exp1_path)}
    exp3 = load_jsonl(exp3_path)
    if not exp1 or not exp3:
        return None

    total = flipped = corrected = damaged = 0
    for r3 in exp3:
        r1 = exp1.get(r3["event_id"])
        if not r1:
            continue
        total  += 1
        gold, p1, p3 = r3["gold"], r1["pred"], r3["final_pred"]
        if p1 != p3:
            flipped += 1
            if p1 != gold and p3 == gold:
                corrected += 1
            elif p1 == gold and p3 != gold:
                damaged += 1

    if total == 0:
        return None
    return {
        "total":           total,
        "flip_rate":       flipped   / total,
        "correction_rate": corrected / total,
        "damage_rate":     damaged   / total,
        "flipped":         flipped,
        "corrected":       corrected,
        "damaged":         damaged,
        "net_gain":        (corrected - damaged) / total,
    }


def print_correction_table(analysis3, analysis3b, out_dir: Path):
    """同时打印 Exp-3 和 Exp-3b 的 correction analysis，并对比 thinking 效果。"""

    def _section(title, analysis):
        lines = [title]
        if analysis:
            lines += [
                f"  Total test events : {analysis['total']}",
                f"  Flip rate         : {analysis['flip_rate']:.1%}  ({analysis['flipped']})",
                f"  Correction rate   : {analysis['correction_rate']:.1%}  ({analysis['corrected']})",
                f"    (Exp-1 wrong → correct)",
                f"  Damage rate       : {analysis['damage_rate']:.1%}  ({analysis['damaged']})",
                f"    (Exp-1 correct → wrong)",
                f"  Net gain          : {analysis['net_gain']:+.1%}",
            ]
        else:
            lines.append("  N/A — predictions not found.")
        return lines

    lines = ["=" * 68,
             "Table 3: Post-Processing Correction Analysis (vs Exp-1)",
             "=" * 68]
    lines += _section("Exp-3  Small+LLM-Post:", analysis3)
    lines.append("")
    lines += _section("Exp-3b Small+LLM-Post(Think):", analysis3b)

    # Thinking 带来的额外增益
    if analysis3 and analysis3b:
        lines += [
            "",
            "Thinking effect (Exp-3b − Exp-3):",
            f"  Net gain delta : {analysis3b['net_gain'] - analysis3['net_gain']:+.1%}",
            f"  Correction Δ   : {analysis3b['correction_rate'] - analysis3['correction_rate']:+.1%}",
            f"  Damage Δ       : {analysis3b['damage_rate']     - analysis3['damage_rate']:+.1%}",
        ]
    lines.append("=" * 68)

    txt = "\n".join(lines)
    print("\n" + txt)
    _write(out_dir / "correction_analysis.txt", txt)

    # markdown
    def _md_block(label, analysis):
        rows = []
        if analysis:
            rows += [
                md_row("Metric", "Rate", "Count"),
                md_sep("l", "r", "r"),
                md_row("Flip rate",       f"{analysis['flip_rate']:.1%}",       analysis["flipped"]),
                md_row("Correction rate", f"{analysis['correction_rate']:.1%}", analysis["corrected"]),
                md_row("Damage rate",     f"{analysis['damage_rate']:.1%}",     analysis["damaged"]),
                md_row("**Net gain**",    f"**{analysis['net_gain']:+.1%}**",   ""),
                "",
                "> *Correction*: Exp-1 wrong → correct  ",
                "> *Damage*: Exp-1 correct → wrong",
            ]
        else:
            rows.append("N/A — predictions not found.")
        return [f"\n### {label}\n"] + rows

    md = ["## Table 3: Correction Analysis (vs Exp-1)\n"]
    md += _md_block("Exp-3 Small+LLM-Post", analysis3)
    md += _md_block("Exp-3b Small+LLM-Post (Thinking)", analysis3b)

    if analysis3 and analysis3b:
        md += [
            "\n### Thinking Effect (Exp-3b − Exp-3)\n",
            md_row("Metric", "Δ"),
            md_sep("l", "r"),
            md_row("Net gain",       f"{analysis3b['net_gain']       - analysis3['net_gain']:+.1%}"),
            md_row("Correction rate",f"{analysis3b['correction_rate']- analysis3['correction_rate']:+.1%}"),
            md_row("Damage rate",    f"{analysis3b['damage_rate']    - analysis3['damage_rate']:+.1%}"),
        ]

    _write(out_dir / "correction_analysis.md", "\n".join(md))
    print(f"Saved -> {out_dir / 'correction_analysis.txt'}  and  correction_analysis.md")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content + "\n")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for all experiments")
    parser.add_argument("--exp1_preds",    default="outputs/exp1_small_only/test_predictions.jsonl")
    parser.add_argument("--exp2_preds",    default="outputs/exp2_llm_pre/test_predictions.jsonl")
    parser.add_argument("--exp3_preds",    default="outputs/exp3_llm_post/test_predictions.jsonl")
    parser.add_argument("--exp3b_preds",   default="outputs/exp3_llm_post/test_predictions_thinking.jsonl",
                        help="Exp-3b: LLM-Post with enable_thinking=True")
    parser.add_argument("--exp2_aug_dir",  default="outputs/exp2_llm_pre",
                        help="Directory with llm_aug_{split}.jsonl for Exp-2 token stats")
    parser.add_argument("--metrics_dir",   default="outputs/metrics")
    args = parser.parse_args()

    out_dir = Path(args.metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Table 1: main results ──────────────────────────────────────────────
    r1  = eval_exp("Exp-1  Small-Only",            Path(args.exp1_preds),  pred_field="pred")
    r2  = eval_exp("Exp-2  LLM-Pre+Small",         Path(args.exp2_preds),  pred_field="pred")
    r3  = eval_exp("Exp-3  Small+LLM-Post",        Path(args.exp3_preds),  pred_field="final_pred")
    r3b = eval_exp("Exp-3b Small+LLM-Post(Think)", Path(args.exp3b_preds), pred_field="final_pred")
    print_main_table(r1, r2, r3, r3b, out_dir)

    # ── Table 2: LLM stats ─────────────────────────────────────────────────
    stats2  = _llm_stats_pre(Path(args.exp2_aug_dir)) \
              if Path(args.exp2_aug_dir).exists() else None
    stats3  = _llm_stats_post(Path(args.exp3_preds)) \
              if Path(args.exp3_preds).exists() else None
    stats3b = _llm_stats_post(Path(args.exp3b_preds)) \
              if Path(args.exp3b_preds).exists() else None
    print_llm_stats_table(stats2, stats3, stats3b, out_dir)

    # ── Table 3: correction analysis ──────────────────────────────────────
    analysis3  = compute_correction_analysis(Path(args.exp1_preds), Path(args.exp3_preds))
    analysis3b = compute_correction_analysis(Path(args.exp1_preds), Path(args.exp3b_preds))
    print_correction_table(analysis3, analysis3b, out_dir)

    # ── Machine-readable dump ─────────────────────────────────────────────
    all_results = {
        "exp1": r1, "exp2": r2, "exp3": r3, "exp3b": r3b,
        "llm_stats_exp2":  stats2,
        "llm_stats_exp3":  stats3,
        "llm_stats_exp3b": stats3b,
        "correction_analysis_exp3":  analysis3,
        "correction_analysis_exp3b": analysis3b,
    }
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved -> {out_dir / 'all_results.json'}")
    print(f"Markdown tables  -> {out_dir}/main_results.md  |  llm_stats.md  |  correction_analysis.md")


if __name__ == "__main__":
    main()
