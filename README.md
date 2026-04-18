# FN_Cascaded: Controlled Variable Experiments on LLM Position in Fake News Detection

A controlled variable experiment framework on the PHEME dataset that compares three configurations with a **fixed small model** and **fixed data split**, varying only where the LLM intervenes:

| Experiment | Description |
|---|---|
| **Exp-1 Small-Only** | DeBERTa-v3-base classifies directly from BasePack — no LLM |
| **Exp-2 LLM-Pre + Small** | LLM pre-processes each event into structured signals; same small model classifies the augmented input |
| **Exp-3 Small + LLM-Post** | Small model produces an initial prediction; LLM acts as final judge given the event + small model report |

**Core research question:** In the same data and small model setting, does the LLM perform better as a *preprocessor* or a *postprocessor*?

---

## System Architecture

```
PHEME rumours (veracity-annotated)
           │
  ┌────────┴────────┐
  │  binary_events  │  (true/false only; unverified dropped)
  └────────┬────────┘
           │
  ┌────────┴────────┐
  │    BasePack     │  source + selected replies + propagation stats
  └────┬────┬───────┘
       │    │
  ─────┘    └──────────────────────────────────────────────┐
  │                                                         │
  ▼                                                         ▼
Exp-1                    Exp-2                           Exp-3
Small Model         LLM Preprocessing              LLM Postprocessing
(no LLM)           BasePack → LLM_Aug             BasePack → Small Model
     │             BasePack+LLM_Aug → Small             SmallReport → LLM
     ▼                    │                                   │
  Fake/True            Fake/True                          Fake/True
```

---

## Project Directory

```
FN_Cascaded/
│
├── README.md
├── configs/
│   └── api_config.yaml              # API key, model, training hyperparams, experiment paths
│
├── prompts/
│   ├── llm_preprocess.txt           # Exp-2: LLM extracts supporting/refuting signals
│   ├── llm_postprocess.txt          # Exp-3: LLM makes final judgment given SmallReport
│   └── rumor_verdict.txt            # Legacy: original routing-based prompt
│
├── data/
│   ├── raw/PHEME/
│   │   └── all-rnr-annotated-threads/
│   │       └── {topic}-all-rnr-threads/
│   │           └── rumours/{tweet_id}/
│   │               ├── source-tweets/{tweet_id}.json
│   │               ├── reactions/*.json
│   │               ├── structure.json
│   │               └── annotation.json      ← veracity label lives here
│   └── processed/
│       ├── binary_events.jsonl              # Fake/True events only
│       ├── train.jsonl / val.jsonl / test.jsonl
│       ├── basepack_train.jsonl             # BasePack for Exp-1 training
│       ├── basepack_val.jsonl
│       ├── basepack_test.jsonl
│       ├── augmented_train.jsonl            # BasePack + LLM_Aug for Exp-2 training
│       ├── augmented_val.jsonl
│       └── augmented_test.jsonl
│
├── baselines/text_cls/
│   └── train.py                     # DeBERTa/RoBERTa fine-tuning (supports --input_field)
│
├── scripts/
│   │── 01_build_pheme_binary.py     # Raw PHEME → binary_events.jsonl (Fake/True only)
│   ├── 02_make_splits.py            # binary_events → train/val/test splits
│   ├── 02_build_basepack.py         # Splits → basepack_{split}.jsonl
│   ├── 03a_train_exp1.sh            # Exp-1: train + infer small model on BasePack
│   ├── 04_predict_small_model.py    # Small model inference (shared by Exp-1 and Exp-2)
│   ├── 05_run_llm_preprocess.py     # Exp-2: call LLM on all splits → llm_aug_{split}.jsonl
│   ├── 06_build_augmented_input.py  # Exp-2: merge BasePack + LLM_Aug
│   ├── 06a_train_exp2.sh            # Exp-2: train + infer small model on augmented input
│   ├── 07_run_llm_postprocess.py    # Exp-3: SmallReport → LLM final judgment
│   ├── 08_eval_all.py               # Unified evaluation: all 3 experiments
│   └── 09_case_study.py             # Case study: where experiments agree/disagree
│
└── outputs/
    ├── exp1_small_only/
    │   ├── model/best_model/            # Saved checkpoint
    │   └── test_predictions.jsonl
    ├── exp2_llm_pre/
    │   ├── llm_aug_{train,val,test}.jsonl   # LLM augmentation outputs
    │   ├── model/best_model/
    │   └── test_predictions.jsonl
    ├── exp3_llm_post/
    │   └── test_predictions.jsonl
    └── metrics/
        ├── main_results.txt             # Table 1: accuracy / macro-F1 / F1-Fake / F1-True
        ├── llm_stats.txt                # Table 2: avg tokens, parse success rate
        ├── correction_analysis.txt      # Table 3: flip/correction/damage rates
        ├── case_study.jsonl             # Qualitative cases
        └── all_results.json             # Machine-readable full results
```

---

## Label Scheme

| Veracity annotation | Label | Class name |
|---|---|---|
| `misinformation=1, true=0` | `1` | **Fake** |
| `misinformation=0, true=1` | `0` | **True** |
| `misinformation=0, true=0` | — | Dropped (unverified) |

Only events in the `rumours/` subdirectory with a valid `annotation.json` are used.

---

## Data Format Reference

### `data/processed/binary_events.jsonl`

```json
{
  "event_id":    "580320995266936832",
  "topic":       "germanwings-crash",
  "veracity":    "true",
  "label":       0,
  "source_text": "BREAKING: Germanwings A320 has crashed ...",
  "replies": [
    {"tweet_id": "580321...", "text": "Oh no ...", "parent": "580320...", "time": "Tue Mar 24 ..."}
  ],
  "structure":   {"580320...": {"580321...": {}}},
  "meta": {"num_replies": 17, "max_depth": 3, "num_branches": 4, "time_span": "..."}
}
```

### `data/processed/basepack_{split}.jsonl`

```json
{
  "event_id":      "580320995266936832",
  "label":         0,
  "basepack_text": "[SOURCE]\nBREAKING: ...\n\n[REPLY_1]\n...\n\n[STATS]\nreply_count=17\n...",
  "source_text":   "BREAKING: ...",
  "selected_replies": ["reply text ...", "..."],
  "stats": {"num_replies": 17, "max_depth": 3, "num_branches": 4, "time_span": "..."}
}
```

### `outputs/exp2_llm_pre/llm_aug_{split}.jsonl`

```json
{
  "event_id": "580320995266936832",
  "label": 0,
  "llm_aug": {
    "claim_summary":      "A Germanwings A320 crashed in the French Alps.",
    "supporting_signals": ["Multiple news outlets confirmed the crash", "..."],
    "refuting_signals":   [],
    "conflict_summary":   "Replies show consensus; no denial observed.",
    "risk_note":          "No suspicious patterns detected."
  },
  "parse_success": true,
  "tokens_used":   312
}
```

### `outputs/exp3_llm_post/test_predictions.jsonl`

```json
{
  "event_id":         "580320995266936832",
  "gold":             0,
  "small_pred":       0,
  "small_pred_label": "True",
  "small_confidence": 0.87,
  "prob_fake":        0.13,
  "prob_true":        0.87,
  "llm_parsed": {
    "final_label":      "True",
    "final_confidence": 0.92,
    "reason":           "Small model prediction is correct. Multiple credible sources confirmed the crash."
  },
  "llm_pred":         0,
  "llm_pred_label":   "True",
  "final_pred":       0,
  "tokens_used":      298
}
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract PHEME data
tar -xjf PHEME/PHEME_veracity.tar.bz2 -C data/raw/PHEME/

# 3. Edit configs/api_config.yaml — set your API key and model
```

---

## Step-by-Step Execution

### Phase 1 — Data Preparation (all experiments share these steps)

```bash
# Step 1: Build binary Fake/True events from PHEME veracity annotations
python scripts/01_build_pheme_binary.py \
    --raw_root data/raw/PHEME/all-rnr-annotated-threads \
    --output   data/processed/binary_events.jsonl

# Step 2: Split into train / val / test (stratified by label)
python scripts/02_make_splits.py \
    --input    data/processed/binary_events.jsonl \
    --strategy random \
    --seed     42

# Step 3: Build BasePack text for each event
python scripts/02_build_basepack.py \
    --input_dir  data/processed \
    --output_dir data/processed \
    --max_replies 8
```

---

### Phase 2 — Exp-1: Small-Only Baseline

```bash
# Train small model on BasePack, then run test inference
bash scripts/03a_train_exp1.sh

# Or with custom settings:
MODEL_NAME=microsoft/deberta-v3-base EPOCHS=5 bash scripts/03a_train_exp1.sh
```

Output: `outputs/exp1_small_only/test_predictions.jsonl`

---

### Phase 3 — Exp-2: LLM as Preprocessor

```bash
# Step 1: Generate LLM augmentation for all splits (train + val + test)
python scripts/05_run_llm_preprocess.py \
    --splits train val test

# Step 2: Merge BasePack + LLM_Aug into augmented input
python scripts/06_build_augmented_input.py \
    --splits train val test

# Step 3: Train same small model on augmented input, then infer
bash scripts/06a_train_exp2.sh
```

Output: `outputs/exp2_llm_pre/test_predictions.jsonl`

> **Note:** The small model architecture and hyperparameters are identical to Exp-1.
> Only the input text changes (BasePack → BasePack + LLM_Aug).

---

### Phase 4 — Exp-3: LLM as Postprocessor

```bash
# Uses Exp-1's trained model and its predictions as SmallReport
python scripts/07_run_llm_postprocess.py \
    --basepack     data/processed/basepack_test.jsonl \
    --predictions  outputs/exp1_small_only/test_predictions.jsonl \
    --output       outputs/exp3_llm_post/test_predictions.jsonl
```

Output: `outputs/exp3_llm_post/test_predictions.jsonl`

> **Note:** No retraining needed — Exp-3 reuses the Exp-1 small model directly.

---

### Phase 5 — Evaluation and Analysis

```bash
# Unified evaluation for all 3 experiments
python scripts/08_eval_all.py

# Case study: interesting examples where experiments differ
python scripts/09_case_study.py
```

---

## Result Tables

After running `08_eval_all.py`, three result files are written to `outputs/metrics/`:

**Table 1 — Main Results** (`main_results.txt`)

| Method | Small Fixed? | LLM Pos | Accuracy | Macro-F1 | F1-Fake | F1-True |
|---|:---:|:---:|---:|---:|---:|---:|
| Exp-1 Small-Only | Yes | None | | | | |
| Exp-2 LLM-Pre + Small | Yes | Pre | | | | |
| Exp-3 Small + LLM-Post | Yes | Post | | | | |

**Table 2 — LLM Cost & Reliability** (`llm_stats.txt`)

| Method | Avg Tokens | Total Tokens | Parse Success |
|---|---:|---:|---:|
| Exp-2 LLM-Pre | | | |
| Exp-3 LLM-Post | | | |

**Table 3 — Post-processing Correction Analysis** (`correction_analysis.txt`)

Tracks for Exp-3 vs Exp-1:

| Metric | Definition |
|---|---|
| **Flip rate** | Fraction of events where Exp-3 changed Exp-1's prediction |
| **Correction rate** | Exp-1 wrong → Exp-3 correct |
| **Damage rate** | Exp-1 correct → Exp-3 wrong |

---

## Configuration Reference

[configs/api_config.yaml](configs/api_config.yaml)

| Key | Default | Description |
|---|---|---|
| `api.model` | `gpt-4.1` | LLM model name |
| `api.temperature` | `0` | Deterministic LLM output |
| `api.max_retries` | `3` | Retries on API error |
| `small_model.model_name` | `microsoft/deberta-v3-base` | HuggingFace backbone |
| `small_model.max_length` | `512` | Token limit (increased for BasePack) |
| `small_model.num_epochs` | `5` | Training epochs |
| `small_model.batch_size` | `16` | Training batch size |
| `small_model.seed` | `42` | Random seed |

---

## Controlled Variable Principle

All three experiments share:
- The same PHEME binary subset (`binary_events.jsonl`)
- The same train / val / test split (same seed)
- The same BasePack preprocessing
- The same small model backbone (`deberta-v3-base`) and hyperparameters

Only the following vary:
- Whether the LLM is involved
- Where the LLM is placed (before or after the small model)
- What the LLM's output is used for (augmenting input vs. overriding final label)

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Small model | DeBERTa-v3-base | Strong text encoder, easy HuggingFace integration |
| Binary task | Fake / True | Controlled setting; unverified label excluded |
| Input representation | BasePack (source + replies + stats) | Unified across all 3 experiments |
| Reply selection | Heuristic (earliest / longest / cross-branch) | No training required, good coverage |
| LLM output format | Structured JSON | Enables automated parsing and error analysis |
| Resume support | Append + skip processed IDs | Safe to interrupt long LLM runs |
| Exp-3 fallback | Use small model pred if LLM parse fails | Ensures 100% coverage |

---

## Extending This System

- Replace heuristic reply selector with a **learned evidence ranker**
- Add **BiGCN** as the small model to leverage propagation graph structure
- Use **batch API** to reduce Exp-2 LLM preprocessing cost
- **Distill** LLM corrections (Exp-3) back into the small model
- Extend to **three-class** veracity (True / False / Unverified) with the full PHEME dataset
