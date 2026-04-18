#!/usr/bin/env bash
# 03_train_small_model.sh
# Train the Stage-1 text classifier (Plan B baseline).
# Run from project root: bash scripts/03_train_small_model.sh

set -e

export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-5}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-256}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/small_model"}

echo "======================================"
echo " Stage-1 small model training"
echo "  model : $MODEL_NAME"
echo "  epochs: $EPOCHS"
echo "  lr    : $LR"
echo "  output: $OUTPUT_DIR"
echo "======================================"

python baselines/text_cls/train.py \
    --train       data/processed/train.jsonl \
    --val         data/processed/val.jsonl \
    --model_name  "$MODEL_NAME" \
    --max_length  "$MAX_LEN" \
    --batch_size  "$BATCH_SIZE" \
    --num_epochs  "$EPOCHS" \
    --lr          "$LR" \
    --output_dir  "$OUTPUT_DIR"

echo "Training done. Model saved to $OUTPUT_DIR/best_model"
