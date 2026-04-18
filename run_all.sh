#!/usr/bin/env bash
# run_all.sh
# End-to-end pipeline runner. Run from project root.
#
# Day 1:
#   bash run_all.sh --stage data
#   bash run_all.sh --stage baseline
#
# Day 2:
#   bash run_all.sh --stage cascade --threshold 0.65
#
# Day 3:
#   bash run_all.sh --stage ablation
#   bash run_all.sh --stage eval_full

set -e

STAGE=${1:-all}
THRESHOLD=${THRESHOLD:-0.65}

# ---- Parse named args ----
for arg in "$@"; do
  case $arg in
    --stage=*) STAGE="${arg#*=}" ;;
    --stage)   shift; STAGE="$1" ;;
    --threshold=*) THRESHOLD="${arg#*=}" ;;
    --threshold)   shift; THRESHOLD="$1" ;;
  esac
done

THR_STR="${THRESHOLD/./}"  # e.g. 0.65 -> 065

echo "========================================"
echo " FN_Cascaded Pipeline"
echo " Stage:     $STAGE"
echo " Threshold: $THRESHOLD"
echo "========================================"

# ---- STAGE: data ----
if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
  echo ""
  echo "[Step 1] Extracting PHEME archive..."
  if [ ! -d "data/raw/PHEME/all-rnr-annotated-threads" ]; then
    tar -xjf PHEME/PHEME_veracity.tar.bz2 -C data/raw/PHEME/
    echo "  Extracted to data/raw/PHEME/"
  else
    echo "  Already extracted, skipping."
  fi

  echo ""
  echo "[Step 2] Building events.jsonl..."
  python scripts/01_build_events.py \
      --raw_root data/raw/PHEME/all-rnr-annotated-threads \
      --output   data/processed/events.jsonl

  echo ""
  echo "[Step 3] Making train/val/test splits..."
  python scripts/02_make_splits.py \
      --input      data/processed/events.jsonl \
      --output_dir data/processed \
      --strategy   random
fi

# ---- STAGE: baseline ----
if [[ "$STAGE" == "baseline" || "$STAGE" == "all" ]]; then
  echo ""
  echo "[Step 4] Training small model..."
  bash scripts/03_train_small_model.sh

  echo ""
  echo "[Step 5] Running inference on test set..."
  python scripts/04_predict_small_model.py \
      --input      data/processed/test.jsonl \
      --model_dir  outputs/small_model/best_model \
      --output     outputs/small_model/test_predictions.jsonl
fi

# ---- STAGE: cascade (single threshold) ----
if [[ "$STAGE" == "cascade" || "$STAGE" == "all" ]]; then
  echo ""
  echo "[Step 6] Packing evidence (threshold=$THRESHOLD)..."
  python scripts/05_pack_evidence.py \
      --events      data/processed/test.jsonl \
      --predictions outputs/small_model/test_predictions.jsonl \
      --threshold   "$THRESHOLD" \
      --output      "outputs/packed_events/packed_${THR_STR}.jsonl"

  echo ""
  echo "[Step 7] Running LLM on routed events..."
  python scripts/06_run_llm.py \
      --input     "outputs/packed_events/packed_${THR_STR}.jsonl" \
      --output    "outputs/llm_outputs/llm_${THR_STR}.jsonl" \
      --threshold "$THRESHOLD"

  echo ""
  echo "[Step 8] Merging predictions..."
  python scripts/07_merge_predictions.py \
      --predictions outputs/small_model/test_predictions.jsonl \
      --threshold   "$THRESHOLD"
fi

# ---- STAGE: ablation (all 3 thresholds) ----
if [[ "$STAGE" == "ablation" ]]; then
  for thr in 0.55 0.65 0.75; do
    thr_str="${thr/./}"
    echo ""
    echo "--- Threshold $thr ---"
    python scripts/05_pack_evidence.py \
        --events      data/processed/test.jsonl \
        --predictions outputs/small_model/test_predictions.jsonl \
        --threshold   "$thr" \
        --output      "outputs/packed_events/packed_${thr_str}.jsonl"

    python scripts/06_run_llm.py \
        --input     "outputs/packed_events/packed_${thr_str}.jsonl" \
        --output    "outputs/llm_outputs/llm_${thr_str}.jsonl" \
        --threshold "$thr"

    python scripts/07_merge_predictions.py \
        --predictions outputs/small_model/test_predictions.jsonl \
        --threshold   "$thr"
  done
fi

# ---- STAGE: eval ----
if [[ "$STAGE" == "eval" ]]; then
  python scripts/08_eval.py --threshold "$THRESHOLD"
fi

if [[ "$STAGE" == "eval_full" || "$STAGE" == "all" ]]; then
  python scripts/08_eval.py --ablation
fi

echo ""
echo "Pipeline complete."
