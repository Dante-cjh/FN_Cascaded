"""
08_eval.py
Evaluate merged predictions and produce all result tables.

Outputs (in outputs/metrics/):
  summary_table.txt   - Table 1: main results (also runs small-model-only for comparison)
  ablation_table.txt  - Table 2: threshold ablation
  error_cases.jsonl   - Table 3: error analysis cases
  case_study.jsonl    - Table 4: selected case studies

Usage:
  # Evaluate a single threshold
  python scripts/08_eval.py --threshold 0.65

  # Run full ablation (0.55, 0.65, 0.75) and generate all tables
  python scripts/08_eval.py --ablation

  # Compare with small-model baseline (no LLM)
  python scripts/08_eval.py --baseline_only
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)


# --------------------------------------------------------------------------- #
#  Loading helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(golds: list[int], preds: list[int]) -> dict:
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro", zero_division=0)
    rumor_recall = recall_score(golds, preds, pos_label=1, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "rumor_recall": rumor_recall,
    }


# --------------------------------------------------------------------------- #
#  Single-threshold evaluation
# --------------------------------------------------------------------------- #

def eval_threshold(threshold: float, args) -> dict | None:
    thr_str = str(threshold).replace(".", "")
    merged_path = Path(f"outputs/metrics/merged_{thr_str}.jsonl")
    llm_path = Path(f"outputs/llm_outputs/llm_{thr_str}.jsonl")

    records = load_jsonl(merged_path)
    if not records:
        print(f"[WARN] No merged records found for threshold {threshold}: {merged_path}")
        return None

    golds = [r["gold"] for r in records]
    final_preds = [r["final_pred"] for r in records]
    small_preds = [r["small_pred"] for r in records]

    final_metrics = compute_metrics(golds, final_preds)
    small_metrics = compute_metrics(golds, small_preds)

    total = len(records)
    used_llm = sum(1 for r in records if r.get("used_llm", False))
    llm_rate = used_llm / total if total else 0

    # Token stats from LLM outputs
    llm_records = load_jsonl(llm_path)
    total_tokens = sum(r.get("tokens_used", 0) for r in llm_records)
    avg_tokens = total_tokens / max(used_llm, 1)

    result = {
        "threshold": threshold,
        "total": total,
        "llm_rate": llm_rate,
        "used_llm": used_llm,
        "avg_tokens": avg_tokens,
        "final": final_metrics,
        "small_only": small_metrics,
    }
    return result


# --------------------------------------------------------------------------- #
#  Table 1: main results
# --------------------------------------------------------------------------- #

def print_main_table(results: list[dict], output_path: Path):
    lines = []
    lines.append("=" * 65)
    lines.append("Table 1: Main Results")
    lines.append("=" * 65)
    header = f"{'Method':<30} {'Accuracy':>9} {'Macro-F1':>9} {'R-Recall':>9}"
    lines.append(header)
    lines.append("-" * 65)

    if results:
        r = results[0]  # Use default threshold result
        lines.append(f"{'Small Model Only':<30} {r['small_only']['accuracy']:>9.4f} "
                     f"{r['small_only']['macro_f1']:>9.4f} "
                     f"{r['small_only']['rumor_recall']:>9.4f}")
        lines.append(f"{'Small Model + LLM Cascade':<30} {r['final']['accuracy']:>9.4f} "
                     f"{r['final']['macro_f1']:>9.4f} "
                     f"{r['final']['rumor_recall']:>9.4f}")
    lines.append("=" * 65)

    table = "\n".join(lines)
    print(table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved -> {output_path}")


# --------------------------------------------------------------------------- #
#  Table 2: threshold ablation
# --------------------------------------------------------------------------- #

def print_ablation_table(results: list[dict], output_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("Table 2: Threshold Ablation")
    lines.append("=" * 70)
    header = f"{'Threshold':>10} {'LLM Rate':>10} {'Avg Tokens':>12} {'Macro-F1':>10} {'R-Recall':>10}"
    lines.append(header)
    lines.append("-" * 70)

    for r in sorted(results, key=lambda x: x["threshold"]):
        lines.append(
            f"{r['threshold']:>10.2f} "
            f"{r['llm_rate']:>10.1%} "
            f"{r['avg_tokens']:>12.0f} "
            f"{r['final']['macro_f1']:>10.4f} "
            f"{r['final']['rumor_recall']:>10.4f}"
        )

    lines.append("=" * 70)
    table = "\n".join(lines)
    print(table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table + "\n")
    print(f"Saved -> {output_path}")


# --------------------------------------------------------------------------- #
#  Table 3: error analysis
# --------------------------------------------------------------------------- #

def analyze_errors(threshold: float, packed_map: dict[str, dict], output_path: Path):
    thr_str = str(threshold).replace(".", "")
    merged_path = Path(f"outputs/metrics/merged_{thr_str}.jsonl")
    records = load_jsonl(merged_path)

    categories = {
        "small_wrong_llm_correct": [],
        "small_correct_llm_wrong": [],
        "both_wrong": [],
        "both_correct": [],
    }

    for r in records:
        gold = r["gold"]
        sp = r["small_pred"]
        fp = r["final_pred"]
        used_llm = r.get("used_llm", False)

        small_correct = (sp == gold)
        final_correct = (fp == gold)

        if used_llm:
            if not small_correct and final_correct:
                categories["small_wrong_llm_correct"].append(r)
            elif small_correct and not final_correct:
                categories["small_correct_llm_wrong"].append(r)
            elif not small_correct and not final_correct:
                categories["both_wrong"].append(r)
        else:
            if small_correct:
                categories["both_correct"].append(r)
            else:
                categories["both_wrong"].append(r)

    print(f"\nError analysis (threshold={threshold}):")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)}")

    # Save detailed error cases (small_wrong_llm_correct + small_correct_llm_wrong)
    error_cases = []
    for cat in ["small_wrong_llm_correct", "small_correct_llm_wrong", "both_wrong"]:
        for r in categories[cat][:20]:  # cap at 20 per category
            r["category"] = cat
            error_cases.append(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ec in error_cases:
            f.write(json.dumps(ec) + "\n")
    print(f"Error cases saved -> {output_path}")

    return categories


# --------------------------------------------------------------------------- #
#  Table 4: case studies
# --------------------------------------------------------------------------- #

def build_case_studies(threshold: float, events_path: Path, llm_path: Path, output_path: Path,
                       n_cases: int = 3):
    thr_str = str(threshold).replace(".", "")
    merged_path = Path(f"outputs/metrics/merged_{thr_str}.jsonl")
    packed_path = Path(f"outputs/packed_events/packed_{thr_str}.jsonl")

    merged = {r["event_id"]: r for r in load_jsonl(merged_path)}
    packed = {e["event_id"]: e for e in load_jsonl(packed_path)}
    llm_results = {r["event_id"]: r for r in load_jsonl(llm_path)}

    # Focus on cases where LLM corrected small model
    good_cases = [
        eid for eid, r in merged.items()
        if r.get("used_llm") and r["small_pred"] != r["gold"] and r["final_pred"] == r["gold"]
    ]

    case_studies = []
    for eid in good_cases[:n_cases]:
        m = merged[eid]
        p = packed.get(eid, {})
        l = llm_results.get(eid, {})

        case_studies.append({
            "event_id": eid,
            "gold": m["gold"],
            "source_text": p.get("source_text", ""),
            "selected_replies": p.get("selected_replies", []),
            "small_model": {
                "pred": m["small_pred"],
                "confidence": m["small_conf"],
            },
            "llm_verdict": l.get("llm_parsed"),
            "final_pred": m["final_pred"],
            "correction": "small_wrong -> LLM_correct",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for cs in case_studies:
            f.write(json.dumps(cs, ensure_ascii=False) + "\n")
    print(f"Case studies saved -> {output_path} ({len(case_studies)} cases)")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation over thresholds [0.55, 0.65, 0.75]")
    parser.add_argument("--events", default="data/processed/test.jsonl")
    parser.add_argument("--metrics_dir", default="outputs/metrics")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [0.55, 0.65, 0.75] if args.ablation else [args.threshold]

    all_results = []
    for thr in thresholds:
        print(f"\n{'='*50}")
        print(f"Evaluating threshold = {thr}")
        r = eval_threshold(thr, args)
        if r:
            all_results.append(r)
            thr_str = str(thr).replace(".", "")
            # Detailed classification report
            merged = load_jsonl(Path(f"outputs/metrics/merged_{thr_str}.jsonl"))
            if merged:
                golds = [r["gold"] for r in merged]
                finals = [r["final_pred"] for r in merged]
                print(classification_report(golds, finals,
                                            target_names=["non-rumour", "rumour"]))

    if all_results:
        default_r = next((r for r in all_results if r["threshold"] == args.threshold),
                         all_results[0])

        print_main_table([default_r], metrics_dir / "summary_table.txt")

        if len(all_results) > 1:
            print_ablation_table(all_results, metrics_dir / "ablation_table.txt")

        # Error analysis & case studies for default threshold
        thr = default_r["threshold"]
        thr_str = str(thr).replace(".", "")
        packed_path = Path(f"outputs/packed_events/packed_{thr_str}.jsonl")
        llm_path = Path(f"outputs/llm_outputs/llm_{thr_str}.jsonl")

        packed_map = {e["event_id"]: e for e in load_jsonl(packed_path)}
        analyze_errors(thr, packed_map, metrics_dir / "error_cases.jsonl")
        build_case_studies(thr, Path(args.events), llm_path, metrics_dir / "case_study.jsonl")

        # Save machine-readable results
        with open(metrics_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved -> {metrics_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
