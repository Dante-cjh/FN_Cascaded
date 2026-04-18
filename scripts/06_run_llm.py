"""
06_run_llm.py
Call the LLM for events routed by the confidence threshold.

Reads packed events (from 05_pack_evidence.py) and calls the API
only for events where route_to_llm == True.

Output: outputs/llm_outputs/llm_{threshold}.jsonl
Each line:
{
  "event_id": str,
  "label": int,
  "llm_raw": str,           # raw API response text
  "llm_parsed": {           # parsed JSON from LLM (or null on parse failure)
    "label": str,           # "rumor" or "non-rumor"
    "confidence": float,
    "evidence_for": [str],
    "evidence_against": [str],
    "propagation_summary": str,
    "final_rationale": str
  },
  "llm_pred": int | null,   # 1=rumor, 0=non-rumor, null if parse failed
  "tokens_used": int
}
"""

import json
import os
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
#  Config helpers
# --------------------------------------------------------------------------- #

def load_config(config_path: str = "configs/api_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env_key = os.environ.get("LLM_API_KEY", "").strip()
    if env_key:
        cfg["api"]["api_key"] = env_key
    if not cfg["api"].get("api_key"):
        raise EnvironmentError(
            "LLM API key is not set.\n"
            "  Option 1: export LLM_API_KEY='sk-...'\n"
            "  Option 2: create a .env file (see .env.example)"
        )
    return cfg


def load_prompt_template(path: str = "prompts/rumor_verdict.txt") -> str:
    with open(path) as f:
        return f.read()


# --------------------------------------------------------------------------- #
#  Prompt rendering
# --------------------------------------------------------------------------- #

def render_prompt(template: str, event: dict) -> str:
    sm = event.get("small_model", {})
    prop = event.get("propagation_summary", {})

    replies_text = "\n".join(
        f"  [{i+1}] {r}" for i, r in enumerate(event.get("selected_replies", []))
    ) or "  (no replies available)"

    sm_pred_str = "rumor" if sm.get("pred", 0) == 1 else "non-rumor"
    sm_conf = f"{sm.get('confidence', 0):.3f}"

    return template.format(
        source_text=event.get("source_text", ""),
        selected_replies=replies_text,
        num_replies=prop.get("num_replies", 0),
        max_depth=prop.get("max_depth", 0),
        num_branches=prop.get("num_branches", 0),
        small_model_pred=sm_pred_str,
        small_model_confidence=sm_conf,
    )


# --------------------------------------------------------------------------- #
#  Response parsing
# --------------------------------------------------------------------------- #

def parse_llm_response(raw: str) -> dict | None:
    """Extract JSON from LLM response. Returns None on failure."""
    # Try to find JSON block (may be wrapped in ```json ... ```)
    clean = raw.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if json_match:
        clean = json_match.group(1)
    else:
        # Find first { ... } block
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end + 1]

    try:
        parsed = json.loads(clean)
        # Validate required fields
        if "label" not in parsed:
            return None
        return parsed
    except json.JSONDecodeError:
        return None


def label_to_int(label_str: str) -> int | None:
    if not label_str:
        return None
    label_str = label_str.lower().strip()
    if "non" in label_str:
        return 0
    if "rumor" in label_str or "rumour" in label_str:
        return 1
    return None


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #

def run_llm(args):
    config = load_config(args.config)
    api_cfg = config["api"]

    client = OpenAI(
        api_key=api_cfg["api_key"],
        base_url=api_cfg["base_url"],
    )
    model = api_cfg["model"]
    temperature = api_cfg.get("temperature", 0)
    max_retries = api_cfg.get("max_retries", 3)
    retry_delay = api_cfg.get("retry_delay", 5)

    prompt_template = load_prompt_template(args.prompt)

    # Load packed events
    with open(args.input) as f:
        all_events = [json.loads(line) for line in f if line.strip()]

    to_process = [e for e in all_events if e.get("route_to_llm", False)]
    print(f"Total packed events:  {len(all_events)}")
    print(f"Routed to LLM:        {len(to_process)}")
    print(f"Model:                {model}")
    print(f"Output:               {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already processed events
    processed_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    processed_ids.add(obj["event_id"])
        print(f"Resuming: {len(processed_ids)} already processed, skipping.")

    remaining = [e for e in to_process if e["event_id"] not in processed_ids]
    print(f"Remaining to process: {len(remaining)}")

    total_tokens = 0
    success = 0
    parse_fail = 0

    with open(output_path, "a") as out:
        for i, event in enumerate(remaining, 1):
            eid = event["event_id"]
            prompt = render_prompt(prompt_template, event)

            # Retry loop
            result = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                    raw_text = resp.choices[0].message.content or ""
                    tokens = resp.usage.total_tokens if resp.usage else 0
                    result = (raw_text, tokens)
                    break
                except Exception as e:
                    print(f"  [attempt {attempt}/{max_retries}] API error for {eid}: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)

            if result is None:
                print(f"  [FAIL] All retries exhausted for {eid}")
                record = {
                    "event_id": eid,
                    "label": event["label"],
                    "llm_raw": None,
                    "llm_parsed": None,
                    "llm_pred": None,
                    "tokens_used": 0,
                }
            else:
                raw_text, tokens = result
                parsed = parse_llm_response(raw_text)
                llm_pred = label_to_int(parsed.get("label") if parsed else None)
                total_tokens += tokens
                if parsed:
                    success += 1
                else:
                    parse_fail += 1
                    print(f"  [PARSE FAIL] event {eid} | raw: {raw_text[:100]}")

                record = {
                    "event_id": eid,
                    "label": event["label"],
                    "llm_raw": raw_text,
                    "llm_parsed": parsed,
                    "llm_pred": llm_pred,
                    "tokens_used": tokens,
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if i % 10 == 0:
                print(f"  Progress: {i}/{len(remaining)} | tokens so far: {total_tokens}")

    avg_tokens = total_tokens / max(success, 1)
    print(f"\nDone.")
    print(f"  Successful:      {success}")
    print(f"  Parse failures:  {parse_fail}")
    print(f"  Total tokens:    {total_tokens}")
    print(f"  Avg tokens/call: {avg_tokens:.0f}")
    print(f"  Output:          {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Packed events file (default: outputs/packed_events/packed_065.jsonl)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="configs/api_config.yaml")
    parser.add_argument("--prompt", default="prompts/rumor_verdict.txt")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Used to auto-name input/output if not specified")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess all events (no resume)")
    args = parser.parse_args()

    thr_str = str(args.threshold).replace(".", "")
    if args.input is None:
        args.input = f"outputs/packed_events/packed_{thr_str}.jsonl"
    if args.output is None:
        args.output = f"outputs/llm_outputs/llm_{thr_str}.jsonl"

    run_llm(args)


if __name__ == "__main__":
    main()
