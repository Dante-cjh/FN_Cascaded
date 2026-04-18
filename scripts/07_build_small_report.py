"""
07_build_small_report.py
Build SmallReport for Exp-3 from Exp-1 predictions + BasePack.

Reads:
  outputs/exp1_small_only/test_predictions.jsonl   (small model predictions)
  data/processed/basepack_test.jsonl               (BasePack for event_summary)

Output: outputs/exp3_llm_post/small_report.jsonl
Each line:
{
  "event_id": str,
  "small_report": {
    "small_pred":       str,    # "Fake" or "True"
    "small_confidence": float,
    "prob_fake":        float,
    "prob_true":        float,
    "event_summary":    str     # rule-based: source prefix + reply count + top replies
  }
}

The event_summary is assembled by rules (no generation model needed):
  - First sentence of source_text (up to 150 chars)
  - reply_count and max_depth from stats
  - Text snippets of the two earliest selected_replies
"""

import json
import argparse
from pathlib import Path


LABEL_NAMES = {0: "True", 1: "Fake"}


# --------------------------------------------------------------------------- #
#  Rule-based event summary
# --------------------------------------------------------------------------- #

def build_event_summary(basepack: dict) -> str:
    source = basepack.get("source_text", "")
    stats  = basepack.get("stats", {})
    replies = basepack.get("selected_replies", [])

    # First sentence / first 150 chars of source
    first_sentence = source.split(".")[0].strip()
    if len(first_sentence) > 150:
        first_sentence = first_sentence[:150] + "…"

    # Propagation info
    prop = (f"reply_count={stats.get('num_replies', '?')}, "
            f"max_depth={stats.get('max_depth', '?')}")

    # Two earliest replies (already ordered chronologically by 02_build_basepack.py)
    reply_snippets = []
    for r in replies[:2]:
        snippet = r.strip()[:80] + ("…" if len(r.strip()) > 80 else "")
        reply_snippets.append(f'"{snippet}"')

    parts = [first_sentence, prop]
    if reply_snippets:
        parts.append("top replies: " + "; ".join(reply_snippets))

    return " | ".join(parts)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def build_small_report(args):
    pred_path     = Path(args.predictions)
    basepack_path = Path(args.basepack)
    output_path   = Path(args.output)

    if not pred_path.exists():
        print(f"ERROR: predictions not found: {pred_path}")
        print("Run Exp-1 first (bash scripts/03a_train_exp1.sh).")
        return
    if not basepack_path.exists():
        print(f"ERROR: basepack not found: {basepack_path}")
        return

    with open(pred_path) as f:
        preds = [json.loads(l) for l in f if l.strip()]
    with open(basepack_path) as f:
        basepacks = {json.loads(l)["event_id"]: json.loads(l)
                     for l in open(basepack_path) if l.strip()}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_bp = 0
    with open(output_path, "w") as out:
        for pred in preds:
            eid  = pred["event_id"]
            bp   = basepacks.get(eid)
            if bp is None:
                missing_bp += 1
                bp = {}

            prob       = pred.get("prob", [0.5, 0.5])
            pred_int   = pred.get("pred", 0)
            confidence = pred.get("confidence", max(prob))

            prob_true = prob[0] if len(prob) > 0 else 0.5
            prob_fake = prob[1] if len(prob) > 1 else 0.5

            record = {
                "event_id": eid,
                "small_report": {
                    "small_pred":       LABEL_NAMES.get(pred_int, str(pred_int)),
                    "small_confidence": round(confidence, 4),
                    "prob_fake":        round(prob_fake, 4),
                    "prob_true":        round(prob_true, 4),
                    "event_summary":    build_event_summary(bp),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"SmallReport built: {written} events -> {output_path}")
    if missing_bp:
        print(f"  [WARN] {missing_bp} events had no matching BasePack entry.")


def main():
    parser = argparse.ArgumentParser(description="Build SmallReport for Exp-3")
    parser.add_argument(
        "--predictions",
        default="outputs/exp1_small_only/test_predictions.jsonl",
        help="Exp-1 small model predictions",
    )
    parser.add_argument(
        "--basepack",
        default="data/processed/basepack_test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/exp3_llm_post/small_report.jsonl",
    )
    args = parser.parse_args()
    build_small_report(args)


if __name__ == "__main__":
    main()
