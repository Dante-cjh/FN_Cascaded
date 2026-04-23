#!/usr/bin/env bash
# 03a_train_exp1.sh
# Exp-1: Train small model on BasePack (Small-Only baseline).
# Run from project root: bash scripts/03a_train_exp1.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/exp1_small_only/model"}

echo "======================================"
echo " Exp-1: Small-Only Training"
echo "  input_mode : base (basepack_text)"
echo "  model      : $MODEL_NAME"
echo "  epochs     : $EPOCHS  lr: $LR  max_len: $MAX_LEN  seed: $SEED"
echo "  output     : $OUTPUT_DIR"
echo "======================================"

python baselines/text_cls/train.py \
    --train       data/processed/basepack_train.jsonl \
    --val         data/processed/basepack_val.jsonl \
    --model_name  "$MODEL_NAME" \
    --max_length  "$MAX_LEN" \
    --batch_size  "$BATCH_SIZE" \
    --num_epochs  "$EPOCHS" \
    --lr          "$LR" \
    --seed        "$SEED" \
    --input_mode  base \
    --label_names True Fake \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "Training done. Running test inference..."

python scripts/04_predict_small_model.py \
    --input       data/processed/basepack_test.jsonl \
    --model_dir   "$OUTPUT_DIR/best_model" \
    --output      outputs/exp1_small_only/test_predictions.jsonl \
    --input_mode  base \
    --label_names True Fake \
    --max_length  "$MAX_LEN"

echo ""
echo "Exp-1 complete."
echo "  Predictions -> outputs/exp1_small_only/test_predictions.jsonl"
echo ""
echo "Next step (Exp-3, no retraining needed):"
echo "  python scripts/07_build_small_report.py"
echo "  python scripts/07_run_llm_postprocess.py"
