#!/usr/bin/env bash
# scripts_Exp2/run_abl1_basepack_only.sh
# Ablation 1: BasePack only (no LLM augmentation)
# Reuses existing data/processed/basepack_{split}.jsonl
# Run from project root: bash scripts_Exp2/run_abl1_basepack_only.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

ABL_NAME="abl1_basepack_only"
MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
OUTPUT_DIR="outputs/exp2_ablation/${ABL_NAME}/model"
PRED_OUT="outputs/exp2_ablation/${ABL_NAME}/test_predictions.jsonl"

echo "======================================"
echo " Exp-2 Ablation: ${ABL_NAME}"
echo "  input_mode : base (basepack_text)"
echo "  model      : $MODEL_NAME"
echo "  epochs     : $EPOCHS  lr: $LR  max_len: $MAX_LEN"
echo "  output     : $OUTPUT_DIR"
echo "======================================"

for split in train val test; do
    if [ ! -f "data/processed/basepack_${split}.jsonl" ]; then
        echo "ERROR: data/processed/basepack_${split}.jsonl not found."
        exit 1
    fi
done

python baselines/text_cls/train.py \
    --train       data/processed/basepack_train.jsonl \
    --val         data/processed/basepack_val.jsonl \
    --model_name  "$MODEL_NAME" \
    --max_length  "$MAX_LEN" \
    --batch_size  "$BATCH_SIZE" \
    --num_epochs  "$EPOCHS" \
    --lr          "$LR" \
    --input_mode  base \
    --label_names True Fake \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "Training done. Running test inference..."

python scripts/04_predict_small_model.py \
    --input       data/processed/basepack_test.jsonl \
    --model_dir   "${OUTPUT_DIR}/best_model" \
    --output      "$PRED_OUT" \
    --input_mode  base \
    --label_names True Fake \
    --max_length  "$MAX_LEN"

echo ""
echo "${ABL_NAME} complete. Predictions -> ${PRED_OUT}"
