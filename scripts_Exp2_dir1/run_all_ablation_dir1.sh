#!/usr/bin/env bash
# scripts_Exp2_dir1/run_all_ablation_dir1.sh
# Run all 8 leave-one-out (w/o) ablation conditions for Dir1 (Verifiability) sequentially.
# Run from project root: bash scripts_Exp2_dir1/run_all_ablation_dir1.sh
#
# To run a single condition:
#   bash scripts_Exp2_dir1/run_dir1_wo_claim_core.sh
#
# To override hyperparameters for all conditions:
#   SEED=0 EPOCHS=10 bash scripts_Exp2_dir1/run_all_ablation_dir1.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

START=$(date +%s)
echo "=============================================="
echo " Exp-2 Dir1 (Verifiability) Leave-One-Out Ablation — 8 Conditions"
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

run_condition "scripts_Exp2_dir1/run_dir1_full.sh"                  "dir1_full                 (all 7 fields)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_claim_core.sh"         "dir1_wo_claim_core        (- claim_core)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_claim_components.sh"   "dir1_wo_claim_components  (- claim_components)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_evidence_basis.sh"     "dir1_wo_evidence_basis    (- evidence_basis)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_verification_gaps.sh"  "dir1_wo_verification_gaps (- verification_gaps)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_source_grounding.sh"   "dir1_wo_source_grounding  (- source_grounding)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_ambiguity_flags.sh"    "dir1_wo_ambiguity_flags   (- ambiguity_flags)"
run_condition "scripts_Exp2_dir1/run_dir1_wo_followup_checks.sh"    "dir1_wo_followup_checks   (- followup_checks)"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "=============================================="
echo " All 8 conditions complete."
echo " Total time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "=============================================="
echo ""
echo "Next: python scripts_Exp2_dir1/eval_ablation_dir1.py"
