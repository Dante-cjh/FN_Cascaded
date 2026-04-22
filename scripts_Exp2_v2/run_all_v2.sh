#!/usr/bin/env bash
# scripts_Exp2_v2/run_all_v2.sh
# Run all 6 leave-one-out (w/o) ablation conditions sequentially.
# Run from project root: bash scripts_Exp2_v2/run_all_v2.sh
#
# To run a single condition:
#   bash scripts_Exp2_v2/run_v2_wo_claim.sh
#
# To override hyperparameters for all conditions:
#   SEED=0 EPOCHS=10 bash scripts_Exp2_v2/run_all_v2.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

START=$(date +%s)
echo "=============================================="
echo " Exp-2 Leave-One-Out Ablation (v2) — 6 Conditions"
echo " Seed: ${SEED:-42}   Epochs: ${EPOCHS:-10}"
echo " $(date)"
echo "=============================================="

run_condition() {
    local script=$1
    local name=$2
    echo ""
    echo "----------------------------------------------"
    echo " Running : ${name}"
    echo " Script  : ${script}"
    echo " Start   : $(date)"
    echo "----------------------------------------------"
    bash "$script"
    echo " Finished: ${name}  ($(date))"
}

run_condition "scripts_Exp2_v2/run_v2_full.sh"         "v2_full         (all 5 fields)"
run_condition "scripts_Exp2_v2/run_v2_wo_claim.sh"     "v2_wo_claim     (- claim_summary)"
run_condition "scripts_Exp2_v2/run_v2_wo_supporting.sh" "v2_wo_supporting (- supporting_signals)"
run_condition "scripts_Exp2_v2/run_v2_wo_refuting.sh"  "v2_wo_refuting  (- refuting_signals)"
run_condition "scripts_Exp2_v2/run_v2_wo_conflict.sh"  "v2_wo_conflict  (- conflict_summary)"
run_condition "scripts_Exp2_v2/run_v2_wo_risk.sh"      "v2_wo_risk      (- risk_note)"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "=============================================="
echo " All 6 conditions complete."
echo " Total time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "=============================================="
echo ""
echo "Next: python scripts_Exp2_v2/eval_ablation_v2.py"
