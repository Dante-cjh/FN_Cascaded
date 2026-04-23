"""
scripts_Exp2_dir1/run_llm_preprocess_dir1.py
Direction 1 (Verifiability): LLM preprocessing with verifiability-focused fields.

Anchor field for parse validation: claim_core

Output fields:
  claim_core, claim_components, evidence_basis, verification_gaps,
  source_grounding, ambiguity_flags, followup_checks

Reads:  data/processed/basepack_{split}.jsonl
Output: outputs/exp2_dir1/llm_aug_{split}.jsonl

Run from project root:
  python scripts_Exp2_dir1/run_llm_preprocess_dir1.py
"""

import json
import os
import re
import time
import argparse
from pathlib import Path

import yaml
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ANCHOR_FIELD = "claim_core"   # required field; parse fails if absent


# --------------------------------------------------------------------------- #
#  Config & prompt
# --------------------------------------------------------------------------- #

def load_config(config_path: str = "configs/api_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
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
            "  Option 2: create a .env file"
        )
    return cfg


def load_prompt_template(path: str) -> str:
    with open(path) as f:
        return f.read()


# --------------------------------------------------------------------------- #
#  Prompt rendering
# --------------------------------------------------------------------------- #

def render_prompt(template: str, event: dict) -> str:
    replies_text = "\n".join(
        f"  [{i+1}] {r}" for i, r in enumerate(event.get("selected_replies", []))
    ) or "  (no replies available)"
    stats = event.get("stats", {})
    return template.format(
        source_text=event.get("source_text", ""),
        selected_replies=replies_text,
        num_replies=stats.get("num_replies", 0),
        max_depth=stats.get("max_depth", 0),
        num_branches=stats.get("num_branches", 0),
    )


# --------------------------------------------------------------------------- #
#  Response parsing
# --------------------------------------------------------------------------- #

def parse_llm_aug(raw: str) -> dict | None:
    clean = raw.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if json_match:
        clean = json_match.group(1)
    else:
        start, end = clean.find("{"), clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end + 1]
    try:
        parsed = json.loads(clean)
        if ANCHOR_FIELD not in parsed:
            return None
        return parsed
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #

def run_llm_preprocess(args):
    config  = load_config(args.config)
    api_cfg = config["api"]

    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"))
    model       = api_cfg["model"]
    temperature = api_cfg.get("temperature", 0)
    max_retries = api_cfg.get("max_retries", 3)
    retry_delay = api_cfg.get("retry_delay", 5)

    prompt_template = load_prompt_template(args.prompt)
    print(f"Prompt  : {args.prompt}")
    print(f"Anchor  : {ANCHOR_FIELD}")

    for split in args.splits:
        input_path  = Path(args.input_dir) / f"basepack_{split}.jsonl"
        output_path = Path(args.output_dir) / f"llm_aug_{split}.jsonl"

        if not input_path.exists():
            print(f"[WARN] Not found: {input_path}, skipping split '{split}'.")
            continue

        with open(input_path) as f:
            all_events = [json.loads(line) for line in f if line.strip()]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        processed_ids: set[str] = set()
        if output_path.exists() and not args.overwrite:
            with open(output_path) as f:
                for line in f:
                    if line.strip():
                        processed_ids.add(json.loads(line)["event_id"])
            print(f"[{split}] Resuming: {len(processed_ids)} already done.")

        remaining = [e for e in all_events if e["event_id"] not in processed_ids]
        print(f"\n[{split}] Total: {len(all_events)} | Remaining: {len(remaining)}")
        print(f"  Model: {model}  |  Output: {output_path}")

        total_tokens = success = parse_fail = 0

        with open(output_path, "a") as out:
            for i, event in enumerate(remaining, 1):
                eid    = event["event_id"]
                prompt = render_prompt(prompt_template, event)
                result = None

                for attempt in range(1, max_retries + 1):
                    try:
                        resp     = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
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
                    record = {"event_id": eid, "label": event["label"],
                              "llm_aug_raw": None, "llm_aug": None,
                              "parse_success": False, "tokens_used": 0}
                else:
                    raw_text, tokens = result
                    parsed = parse_llm_aug(raw_text)
                    total_tokens += tokens
                    if parsed:
                        success  += 1; parse_ok = True
                    else:
                        parse_fail += 1; parse_ok = False
                        print(f"  [PARSE FAIL] {eid} | raw: {raw_text[:80]}")
                    record = {"event_id": eid, "label": event["label"],
                              "llm_aug_raw": raw_text, "llm_aug": parsed,
                              "parse_success": parse_ok, "tokens_used": tokens}

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                if i % 20 == 0:
                    print(f"  [{split}] {i}/{len(remaining)} | tokens: {total_tokens}")

        total = success + parse_fail
        print(f"\n[{split}] Done. Success: {success}/{total} | "
              f"Avg tokens: {total_tokens // max(success, 1)}")


def main():
    parser = argparse.ArgumentParser(
        description="Dir1 (Verifiability): Run LLM preprocessing"
    )
    parser.add_argument("--input_dir",  default="data/processed")
    parser.add_argument("--output_dir", default="outputs/exp2_dir1")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--config",  default="configs/api_config.yaml")
    parser.add_argument("--prompt",  default="prompts/llm_preprocess_dir1.txt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_llm_preprocess(args)


if __name__ == "__main__":
    main()
