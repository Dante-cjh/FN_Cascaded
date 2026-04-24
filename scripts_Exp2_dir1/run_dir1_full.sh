#!/usr/bin/env bash
# scripts_Exp2_dir1/run_dir1_full.sh
# W/O ablation baseline: full LLM aug (all 7 Dir1 fields)
# Run from project root: bash scripts_Exp2_dir1/run_dir1_full.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

ABL_MODE="full"
ABL_NAME="dir1_full"
MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
SEED=${SEED:-42}
DATA_DIR="data/processed/exp2_ablation_dir1"
OUTPUT_DIR="outputs/exp2_ablation_dir1/${ABL_NAME}/model"
PRED_OUT="outputs/exp2_ablation_dir1/${ABL_NAME}/test_predictions.jsonl"

echo "======================================"
echo " Exp-2 Dir1 W/O Ablation: ${ABL_NAME}"
echo "  removed    : (none — full 7 fields)"
echo "  fields     : claim_core + claim_components + evidence_basis +"
echo "               verification_gaps + source_grounding + ambiguity_flags + followup_checks"
echo "  model      : $MODEL_NAME  seed: $SEED"
echo "  epochs     : $EPOCHS  lr: $LR  max_len: $MAX_LEN"
echo "  output     : $OUTPUT_DIR"
echo "======================================"

echo "Step 1/3: Building dataset (${ABL_MODE})..."
python scripts_Exp2_dir1/build_ablation_dir1_dataset.py \
    --ablation_mode "$ABL_MODE" \
    --output_dir    "$DATA_DIR"

echo ""
echo "Step 2/3: Training..."
python baselines/text_cls/train.py \
    --train       "${DATA_DIR}/${ABL_MODE}_train.jsonl" \
    --val         "${DATA_DIR}/${ABL_MODE}_val.jsonl" \
    --model_name  "$MODEL_NAME" \
    --max_length  "$MAX_LEN" \
    --batch_size  "$BATCH_SIZE" \
    --num_epochs  "$EPOCHS" \
    --lr          "$LR" \
    --seed        "$SEED" \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "Step 3/3: Test inference..."
python scripts/04_predict_small_model.py \
    --input       "${DATA_DIR}/${ABL_MODE}_test.jsonl" \
    --model_dir   "${OUTPUT_DIR}/best_model" \
    --output      "$PRED_OUT" \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --max_length  "$MAX_LEN"

echo ""
echo "${ABL_NAME} complete. Predictions -> ${PRED_OUT}"
