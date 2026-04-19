"""
07_run_llm_postprocess.py
Exp-3: Small model first, LLM as final judge (postprocessor).

Pipeline:
  1. Run 07_build_small_report.py to produce small_report.jsonl
  2. Run THIS script: reads BasePack + SmallReport, calls LLM, writes final predictions

Reads:
  data/processed/basepack_test.jsonl              (original BasePack)
  outputs/exp3_llm_post/small_report.jsonl        (from 07_build_small_report.py)

Output: outputs/exp3_llm_post/test_predictions.jsonl
Each line:
{
  "event_id":         str,
  "gold":             int,
  "small_pred":       int,
  "small_pred_label": str,   # "Fake" or "True"
  "small_confidence": float,
  "prob_fake":        float,
  "prob_true":        float,
  "llm_raw":          str,
  "llm_parsed": {
    "final_label":      str,
    "final_confidence": float,
    "reason":           str
  },
  "llm_pred":         int,   # 1=Fake, 0=True, -1=parse failed
  "llm_pred_label":   str,
  "final_pred":       int    # llm_pred if parse succeeded, else small_pred
  "tokens_used":      int
}
"""

import json
import re
import time
import argparse
from pathlib import Path

import os

import yaml
from openai import OpenAI

# Optional: load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------- #
#  Config & prompt
# --------------------------------------------------------------------------- #

def load_config(config_path: str = "configs/api_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # 环境变量优先级高于 yaml 配置，同时兼容 DASHSCOPE_API_KEY 和 LLM_API_KEY
    env_key = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    if env_key:
        cfg["api"]["api_key"] = env_key
    if not cfg["api"].get("api_key"):
        raise EnvironmentError(
            "LLM API key is not set.\n"
            "  Option 1: export DASHSCOPE_API_KEY='sk-...'\n"
            "  Option 2: create a .env file (see .env.example)"
        )
    return cfg


def load_prompt_template(path: str = "prompts/llm_postprocess.txt") -> str:
    with open(path) as f:
        return f.read()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
#  Prompt rendering
# --------------------------------------------------------------------------- #

def render_prompt(template: str, basepack: dict, small_report: dict) -> str:
    replies_text = "\n".join(
        f"  [{i+1}] {r}" for i, r in enumerate(basepack.get("selected_replies", []))
    ) or "  (no replies available)"

    stats = basepack.get("stats", {})
    return template.format(
        source_text      = basepack.get("source_text", ""),
        selected_replies = replies_text,
        num_replies      = stats.get("num_replies", 0),
        max_depth        = stats.get("max_depth", 0),
        num_branches     = stats.get("num_branches", 0),
        small_pred       = small_report.get("small_pred", "Unknown"),
        small_confidence = small_report.get("small_confidence", 0.5),
        prob_fake        = small_report.get("prob_fake", 0.5),
        prob_true        = small_report.get("prob_true", 0.5),
        event_summary    = small_report.get("event_summary", ""),
    )


# --------------------------------------------------------------------------- #
#  Response parsing
# --------------------------------------------------------------------------- #

def parse_llm_post(raw: str) -> dict | None:
    clean = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if m:
        clean = m.group(1)
    else:
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1:
            clean = clean[s:e + 1]
    try:
        parsed = json.loads(clean)
        return parsed if "final_label" in parsed else None
    except json.JSONDecodeError:
        return None


def final_label_to_int(label_str: str) -> int:
    """'Fake' → 1, 'True' → 0, unknown → -1."""
    if not label_str:
        return -1
    s = label_str.strip().lower()
    if s in ("fake", "false", "misinformation"):
        return 1
    if s in ("true", "real", "verified"):
        return 0
    return -1


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #

def run_llm_postprocess(args):
    config  = load_config(args.config)
    api_cfg = config["api"]

    client = OpenAI(
        api_key  = api_cfg["api_key"],
        base_url = api_cfg["base_url"],
    )
    model_name  = api_cfg["model"]
    temperature = api_cfg.get("temperature", 0)
    max_retries = api_cfg.get("max_retries", 3)
    retry_delay = api_cfg.get("retry_delay", 5)

    prompt_template = load_prompt_template(args.prompt)

    # ── Load data ─────────────────────────────────────────────────────────── #
    basepack_path     = Path(args.basepack)
    small_report_path = Path(args.small_report)
    exp1_preds_path   = Path(args.exp1_preds)
    output_path       = Path(args.output)

    if not basepack_path.exists():
        print(f"ERROR: BasePack not found: {basepack_path}")
        return

    basepacks = {e["event_id"]: e for e in load_jsonl(basepack_path)}

    # SmallReport: prefer dedicated file; fall back to Exp-1 predictions
    small_reports: dict[str, dict] = {}
    if small_report_path.exists():
        for rec in load_jsonl(small_report_path):
            small_reports[rec["event_id"]] = rec["small_report"]
        print(f"Loaded {len(small_reports)} SmallReports from {small_report_path}")
    elif exp1_preds_path.exists():
        print(f"[WARN] {small_report_path} not found. "
              f"Building SmallReport inline from {exp1_preds_path}.")
        print("Tip: run  python scripts/07_build_small_report.py  first.")
        for pred in load_jsonl(exp1_preds_path):
            prob = pred.get("prob", [0.5, 0.5])
            pred_int = pred.get("pred", 0)
            small_reports[pred["event_id"]] = {
                "small_pred":       "Fake" if pred_int == 1 else "True",
                "small_confidence": round(pred.get("confidence", max(prob)), 4),
                "prob_fake":        round(prob[1] if len(prob) > 1 else 0.5, 4),
                "prob_true":        round(prob[0] if len(prob) > 0 else 0.5, 4),
                "event_summary":    pred.get("source_text", "")[:150],
            }
    else:
        print(f"ERROR: Neither {small_report_path} nor {exp1_preds_path} found.")
        print("Run Exp-1 (bash scripts/03a_train_exp1.sh) then "
              "07_build_small_report.py first.")
        return

    # We need gold labels: get them from Exp-1 predictions or BasePack
    gold_map: dict[str, int] = {}
    if exp1_preds_path.exists():
        for r in load_jsonl(exp1_preds_path):
            gold_map[r["event_id"]] = r["gold"]
    else:
        for eid, bp in basepacks.items():
            gold_map[eid] = bp.get("label", -1)

    # The event list to process = intersection of basepacks and small_reports
    event_ids = [eid for eid in small_reports if eid in basepacks]
    print(f"Events to process: {len(event_ids)}")
    print(f"Model: {model_name}  |  Output: {output_path}")

    # ── Resume support ─────────────────────────────────────────────────────── #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        for line in open(output_path):
            if line.strip():
                processed_ids.add(json.loads(line)["event_id"])
        print(f"Resuming: {len(processed_ids)} already done.")

    remaining = [eid for eid in event_ids if eid not in processed_ids]
    print(f"Remaining: {len(remaining)}")

    total_tokens = 0
    llm_success  = 0
    llm_fail     = 0
    parse_fail   = 0

    with open(output_path, "a") as out:
        for i, eid in enumerate(remaining, 1):
            bp           = basepacks[eid]
            small_report = small_reports[eid]
            gold         = gold_map.get(eid, -1)

            small_pred_label = small_report["small_pred"]
            small_pred_int   = 1 if small_pred_label == "Fake" else 0

            prompt = render_prompt(prompt_template, bp, small_report)

            result = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model       = model_name,
                        messages    = [{"role": "user", "content": prompt}],
                        temperature = temperature,
                    )
                    raw_text = resp.choices[0].message.content or ""
                    tokens   = resp.usage.total_tokens if resp.usage else 0
                    result   = (raw_text, tokens)
                    break
                except Exception as e:
                    print(f"  [attempt {attempt}/{max_retries}] API error for {eid}: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)

            if result is None:
                print(f"  [FAIL] All retries exhausted for {eid}")
                record = {
                    "event_id":         eid,
                    "gold":             gold,
                    "small_pred":       small_pred_int,
                    "small_pred_label": small_pred_label,
                    "small_confidence": small_report["small_confidence"],
                    "prob_fake":        small_report["prob_fake"],
                    "prob_true":        small_report["prob_true"],
                    "llm_raw":          None,
                    "llm_parsed":       None,
                    "llm_pred":         -1,
                    "llm_pred_label":   "N/A",
                    "final_pred":       small_pred_int,
                    "tokens_used":      0,
                }
                llm_fail += 1
            else:
                raw_text, tokens = result
                parsed           = parse_llm_post(raw_text)
                total_tokens    += tokens

                if parsed:
                    llm_pred_int   = final_label_to_int(parsed.get("final_label", ""))
                    llm_pred_label = parsed.get("final_label", "N/A")
                    final_pred     = llm_pred_int if llm_pred_int >= 0 else small_pred_int
                    llm_success   += 1
                else:
                    llm_pred_int   = -1
                    llm_pred_label = "N/A"
                    final_pred     = small_pred_int
                    parse_fail    += 1
                    print(f"  [PARSE FAIL] {eid} | raw: {raw_text[:80]}")

                record = {
                    "event_id":         eid,
                    "gold":             gold,
                    "small_pred":       small_pred_int,
                    "small_pred_label": small_pred_label,
                    "small_confidence": small_report["small_confidence"],
                    "prob_fake":        small_report["prob_fake"],
                    "prob_true":        small_report["prob_true"],
                    "llm_raw":          raw_text,
                    "llm_parsed":       parsed,
                    "llm_pred":         llm_pred_int,
                    "llm_pred_label":   llm_pred_label,
                    "final_pred":       final_pred,
                    "tokens_used":      tokens,
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if i % 10 == 0:
                print(f"  Progress: {i}/{len(remaining)} | tokens: {total_tokens}")

    print(f"\nExp-3 done.")
    print(f"  LLM success   : {llm_success}")
    print(f"  Parse failures: {parse_fail}")
    print(f"  API failures  : {llm_fail}")
    print(f"  Total tokens  : {total_tokens}")
    print(f"  Avg tokens/req: {total_tokens / max(llm_success, 1):.0f}")
    print(f"  Output        : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp-3: LLM postprocessing")
    parser.add_argument(
        "--basepack",
        default="data/processed/basepack_test.jsonl",
    )
    parser.add_argument(
        "--small_report",
        default="outputs/exp3_llm_post/small_report.jsonl",
        help="SmallReport file from 07_build_small_report.py (preferred input)",
    )
    parser.add_argument(
        "--exp1_preds",
        default="outputs/exp1_small_only/test_predictions.jsonl",
        help="Fallback: Exp-1 predictions used if small_report.jsonl is absent",
    )
    parser.add_argument(
        "--output",
        default="outputs/exp3_llm_post/test_predictions.jsonl",
    )
    parser.add_argument("--config",    default="configs/api_config.yaml")
    parser.add_argument("--prompt",    default="prompts/llm_postprocess.txt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_llm_postprocess(args)


if __name__ == "__main__":
    main()
