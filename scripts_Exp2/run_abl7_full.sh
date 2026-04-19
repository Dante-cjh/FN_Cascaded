#!/usr/bin/env bash
# scripts_Exp2/run_abl7_full.sh
# Ablation 7: BasePack + full LLM augmentation (all 5 fields)
# Reuses existing data/processed/augmented_{split}.jsonl (same as Exp-2 main)
# Run from project root: bash scripts_Exp2/run_abl7_full.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

ABL_NAME="abl7_full_llm_aug"
MODEL_NAME=${MODEL_NAME:-"microsoft/deberta-v3-base"}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-5}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-512}
OUTPUT_DIR="outputs/exp2_ablation/${ABL_NAME}/model"
PRED_OUT="outputs/exp2_ablation/${ABL_NAME}/test_predictions.jsonl"

echo "======================================"
echo " Exp-2 Ablation: ${ABL_NAME}"
echo "  fields     : claim_summary + supporting + refuting + conflict_summary + risk_note"
echo "  input_mode : base_plus_llm_aug (augmented_text)"
echo "  model      : $MODEL_NAME"
echo "  epochs     : $EPOCHS  lr: $LR  max_len: $MAX_LEN"
echo "  output     : $OUTPUT_DIR"
echo "======================================"

for split in train val test; do
    if [ ! -f "data/processed/augmented_${split}.jsonl" ]; then
        echo "ERROR: data/processed/augmented_${split}.jsonl not found."
        echo "Run first: python scripts/05_run_llm_preprocess.py"
        echo "           python scripts/06_build_augmented_dataset.py"
        exit 1
    fi
done

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
    --model_dir   "${OUTPUT_DIR}/best_model" \
    --output      "$PRED_OUT" \
    --input_mode  base_plus_llm_aug \
    --label_names True Fake \
    --max_length  "$MAX_LEN"

echo ""
echo "${ABL_NAME} complete. Predictions -> ${PRED_OUT}"
