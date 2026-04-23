"""
scripts_Exp2_dir2/eval_dir2.py
Direction 2 (Narrative/Manipulation): Evaluate test predictions.

Reads:  outputs/exp2_dir2/test_predictions.jsonl
Writes: outputs/exp2_dir2/metrics/results.{txt,json}

Run from project root:
  python scripts_Exp2_dir2/eval_dir2.py
"""

import json
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

EXP_NAME    = "Dir2 (Narrative/Manipulation)"
PRED_PATH   = "outputs/exp2_dir2/test_predictions.jsonl"
METRICS_DIR = "outputs/exp2_dir2/metrics"

FIELDS = [
    "narrative_frame", "persuasion_cues", "engagement_pattern",
    "coordination_signals", "evidence_visibility",
    "attention_triggers", "manipulation_risk_profile",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",       default=PRED_PATH)
    parser.add_argument("--output_dir", default=METRICS_DIR)
    args = parser.parse_args()

    pred_path  = Path(args.pred)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        print(f"[ERROR] Predictions not found: {pred_path}")
        return

    golds, preds = [], []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                golds.append(rec["gold"])
                preds.append(rec["pred"])

    acc      = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro",    zero_division=0)
    f1_true  = f1_score(golds, preds, pos_label=0, average="binary", zero_division=0)
    f1_fake  = f1_score(golds, preds, pos_label=1, average="binary", zero_division=0)
    report   = classification_report(golds, preds, target_names=["True", "Fake"],
                                     zero_division=0)

    summary = (
        f"Experiment : {EXP_NAME}\n"
        f"Prompt     : prompts/llm_preprocess_dir2.txt\n"
        f"Fields     : {', '.join(FIELDS)}\n"
        f"N (test)   : {len(golds)}\n"
        f"\n"
        f"Accuracy   : {acc:.4f}  ({acc*100:.2f}%)\n"
        f"Macro-F1   : {macro_f1:.4f}  ({macro_f1*100:.2f}%)\n"
        f"F1-True    : {f1_true:.4f}\n"
        f"F1-Fake    : {f1_fake:.4f}\n"
        f"\nClassification Report:\n{report}"
    )

    print(summary)
    (output_dir / "results.txt").write_text(summary)

    metrics = {
        "experiment": EXP_NAME,
        "n": len(golds),
        "accuracy": acc, "macro_f1": macro_f1,
        "f1_true": f1_true, "f1_fake": f1_fake,
        "fields": FIELDS,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved -> {output_dir}")


if __name__ == "__main__":
    main()
