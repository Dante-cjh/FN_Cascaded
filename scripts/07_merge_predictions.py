"""
07_merge_predictions.py
Merge small-model predictions and LLM outputs into a single final prediction file.

Logic:
  if small_model_confidence >= threshold:
      final_pred = small_model_pred
  else:
      final_pred = llm_pred  (fallback to small_model_pred if LLM failed)

Output: outputs/metrics/merged_{threshold}.jsonl
Each line:
{
  "event_id":   str,
  "gold":       int,
  "small_pred": int,
  "small_conf": float,
  "used_llm":   bool,
  "llm_pred":   int | null,
  "final_pred": int
}
"""

import json
import argparse
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def merge(args):
    threshold = args.threshold
    thr_str = str(threshold).replace(".", "")

    # Resolve paths
    preds_path = Path(args.predictions)
    packed_path = Path(args.packed) if args.packed else Path(f"outputs/packed_events/packed_{thr_str}.jsonl")
    llm_path = Path(args.llm) if args.llm else Path(f"outputs/llm_outputs/llm_{thr_str}.jsonl")
    output_path = Path(args.output) if args.output else Path(f"outputs/metrics/merged_{thr_str}.jsonl")

    # Load
    preds = {p["event_id"]: p for p in load_jsonl(preds_path)}
    packed = {e["event_id"]: e for e in load_jsonl(packed_path)}

    llm_map: dict[str, dict] = {}
    if llm_path.exists():
        for r in load_jsonl(llm_path):
            llm_map[r["event_id"]] = r
    else:
        print(f"[WARN] LLM output file not found: {llm_path}. All events will use small model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    used_llm_count = 0
    llm_fail_count = 0

    for eid, pred in preds.items():
        packed_event = packed.get(eid, {})
        route_to_llm = packed_event.get("route_to_llm", False)
        small_conf = pred["confidence"]
        small_pred = pred["pred"]

        used_llm = False
        llm_pred = None
        final_pred = small_pred

        if route_to_llm:
            llm_record = llm_map.get(eid)
            if llm_record and llm_record.get("llm_pred") is not None:
                llm_pred = llm_record["llm_pred"]
                final_pred = llm_pred
                used_llm = True
                used_llm_count += 1
            else:
                # LLM failed or not called; fall back to small model
                final_pred = small_pred
                llm_fail_count += 1

        records.append({
            "event_id": eid,
            "gold": pred["gold"],
            "small_pred": small_pred,
            "small_conf": small_conf,
            "used_llm": used_llm,
            "llm_pred": llm_pred,
            "final_pred": final_pred,
        })

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    total = len(records)
    routed = sum(1 for r in records if packed.get(r["event_id"], {}).get("route_to_llm", False))
    print(f"Merged {total} events -> {output_path}")
    print(f"  Threshold:          {threshold}")
    print(f"  Routed to LLM:      {routed} ({routed/total:.1%})")
    print(f"  Used LLM:           {used_llm_count}")
    print(f"  LLM fallback (fail):{llm_fail_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/small_model/test_predictions.jsonl")
    parser.add_argument("--packed", default=None, help="Packed events file (auto-named if omitted)")
    parser.add_argument("--llm", default=None, help="LLM outputs file (auto-named if omitted)")
    parser.add_argument("--output", default=None, help="Output path (auto-named if omitted)")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
