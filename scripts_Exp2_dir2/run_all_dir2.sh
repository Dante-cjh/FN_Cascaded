#!/usr/bin/env bash
# scripts_Exp2_dir2/run_all_dir2.sh
# Direction 2 (Narrative/Manipulation): Full pipeline
# Run from project root: bash scripts_Exp2_dir2/run_all_dir2.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
SEED=${SEED:-42}
LLM_OUT="outputs/exp2_dir2"
DATA_OUT="data/processed/exp2_dir2"
MODEL_OUT="outputs/exp2_dir2/model"
PRED_OUT="outputs/exp2_dir2/test_predictions.jsonl"

echo "=============================================="
echo " Exp-2 Direction 2 (Narrative/Manipulation)"
echo " Prompt: prompts/llm_preprocess_dir2.txt"
echo " Fields: narrative_frame / persuasion_cues / engagement_pattern /"
echo "         coordination_signals / evidence_visibility /"
echo "         attention_triggers / manipulation_risk_profile"
echo " model=$MODEL_NAME  seed=$SEED  epochs=$EPOCHS"
echo "=============================================="

echo ""
echo "Step 1/4: LLM preprocessing..."
python scripts_Exp2_dir2/run_llm_preprocess_dir2.py \
    --output_dir "$LLM_OUT"

echo ""
echo "Step 2/4: Building augmented dataset..."
python scripts_Exp2_dir2/build_dataset_dir2.py \
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
echo "Dir2 pipeline complete. Predictions -> ${PRED_OUT}"
echo "Evaluate: python scripts_Exp2_dir2/eval_dir2.py"
