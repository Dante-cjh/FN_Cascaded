#!/usr/bin/env bash
# scripts_Exp2_dir1/run_all_dir1.sh
# Direction 1 (Verifiability): Full pipeline — LLM preprocess → build dataset → train → infer
# Run from project root: bash scripts_Exp2_dir1/run_all_dir1.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
SEED=${SEED:-42}
LLM_OUT="outputs/exp2_dir1"
DATA_OUT="data/processed/exp2_dir1"
MODEL_OUT="outputs/exp2_dir1/model"
PRED_OUT="outputs/exp2_dir1/test_predictions.jsonl"

echo "=============================================="
echo " Exp-2 Direction 1 (Verifiability)"
echo " Prompt: prompts/llm_preprocess_dir1.txt"
echo " Fields: claim_core / claim_components / evidence_basis /"
echo "         verification_gaps / source_grounding /"
echo "         ambiguity_flags / followup_checks"
echo " model=$MODEL_NAME  seed=$SEED  epochs=$EPOCHS"
echo "=============================================="

echo ""
echo "Step 1/4: LLM preprocessing (generating llm_aug)..."
python scripts_Exp2_dir1/run_llm_preprocess_dir1.py \
    --output_dir "$LLM_OUT"

echo ""
echo "Step 2/4: Building augmented dataset..."
python scripts_Exp2_dir1/build_dataset_dir1.py \
    --llm_aug_dir "$LLM_OUT" \
    --output_dir  "$DATA_OUT"

echo ""
echo "Step 3/4: Training small model..."
python baselines/text_cls/train.py \
    --train       "${DATA_OUT}/augmented_train.jsonl" \
    --val         "${DATA_OUT}/augmented_val.jsonl" \
    --model_name  "$MODEL_NAME" \
    --max_length  "$MAX_LEN" \
    --batch_size  "$BATCH_SIZE" \
    --num_epochs  "$EPOCHS" \
    --lr          "$LR" \
    --seed        "$SEED" \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --output_dir  "$MODEL_OUT"

echo ""
echo "Step 4/4: Test inference..."
python scripts/04_predict_small_model.py \
    --input       "${DATA_OUT}/augmented_test.jsonl" \
    --model_dir   "${MODEL_OUT}/best_model" \
    --output      "$PRED_OUT" \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --max_length  "$MAX_LEN"

echo ""
echo "Dir1 pipeline complete. Predictions -> ${PRED_OUT}"
echo "Evaluate: python scripts_Exp2_dir1/eval_dir1.py"
