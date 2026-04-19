#!/usr/bin/env bash
# scripts_Exp2/run_all_ablations.sh
# Run all 7 Exp-2 ablation conditions sequentially.
# Run from project root: bash scripts_Exp2/run_all_ablations.sh
#
# To run a single condition instead:
#   bash scripts_Exp2/run_abl2_claim_only.sh
#
# To override hyperparameters for all conditions:
#   MODEL_NAME=... EPOCHS=3 bash scripts_Exp2/run_all_ablations.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

START=$(date +%s)
echo "=============================================="
echo " Exp-2 Ablation Study — All 7 Conditions"
echo " $(date)"
echo "=============================================="
echo ""

run_condition() {
    local script=$1
    local name=$2
    echo ""
    echo "----------------------------------------------"
    echo " Running: ${name}"
    echo " Script : ${script}"
    echo " Start  : $(date)"
    echo "----------------------------------------------"
    bash "$script"
    echo " Finished: ${name}  ($(date))"
}

run_condition "scripts_Exp2/run_abl1_basepack_only.sh"  "abl1: BasePack only"
run_condition "scripts_Exp2/run_abl2_claim_only.sh"     "abl2: BasePack + claim_summary"
run_condition "scripts_Exp2/run_abl3_claim_signals.sh"  "abl3: BasePack + claim + supporting/refuting"
run_condition "scripts_Exp2/run_abl4_signals_only.sh"   "abl4: BasePack + supporting/refuting only"
run_condition "scripts_Exp2/run_abl5_analysis_only.sh"  "abl5: BasePack + conflict_summary + risk_note"
run_condition "scripts_Exp2/run_abl6_claim_analysis.sh" "abl6: BasePack + claim + conflict_summary + risk_note"
run_condition "scripts_Exp2/run_abl7_full.sh"           "abl7: BasePack + full LLM aug"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "=============================================="
echo " All 7 conditions complete."
echo " Total time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "=============================================="
echo ""
echo "Next: python scripts_Exp2/eval_ablation.py"
