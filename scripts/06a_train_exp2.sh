#!/usr/bin/env bash
# 06a_train_exp2.sh
# Exp-2: Train small model on LLM-augmented BasePack (LLM-Pre + Small).
# Assumes augmented_{train,val,test}.jsonl already built by 06_build_augmented_dataset.py.
# Run from project root: bash scripts/06a_train_exp2.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/exp2_llm_pre/model"}

echo "======================================"
echo " Exp-2: LLM-Pre + Small Training"
echo "  input_mode : base_plus_llm_aug (augmented_text)"
echo "  model      : $MODEL_NAME"
echo "  epochs     : $EPOCHS  lr: $LR  max_len: $MAX_LEN"
echo "  output     : $OUTPUT_DIR"
echo "======================================"

# Sanity check: augmented files must exist
if [ ! -f "data/processed/augmented_train.jsonl" ]; then
    echo "ERROR: augmented_train.jsonl not found."
    echo "Run first: python scripts/05_run_llm_preprocess.py"
    echo "           python scripts/06_build_augmented_dataset.py"
    exit 1
fi

python baselines/text_cls/train.py \
    --train       data/processed/augmented_train.jsonl \
    --val         data/processed/augmented_val.jsonl \
    --model_name  "$MODEL_NAME" \
    --max_length  "$MAX_LEN" \
    --batch_size  "$BATCH_SIZE" \
    --num_epochs  "$EPOCHS" \
    --lr          "$LR" \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "Training done. Running test inference..."

python scripts/04_predict_small_model.py \
    --input       data/processed/augmented_test.jsonl \
    --model_dir   "$OUTPUT_DIR/best_model" \
    --output      outputs/exp2_llm_pre/test_predictions.jsonl \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --max_length  "$MAX_LEN"

echo ""
echo "Exp-2 complete."
echo "  Predictions -> outputs/exp2_llm_pre/test_predictions.jsonl"
