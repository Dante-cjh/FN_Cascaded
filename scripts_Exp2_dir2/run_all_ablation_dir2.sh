#!/usr/bin/env bash
# scripts_Exp2_dir2/run_all_ablation_dir2.sh
# Run all 8 leave-one-out (w/o) ablation conditions for Dir2 (Narrative/Manipulation) sequentially.
# Run from project root: bash scripts_Exp2_dir2/run_all_ablation_dir2.sh
#
# To run a single condition:
#   bash scripts_Exp2_dir2/run_dir2_wo_narrative_frame.sh
#
# To override hyperparameters for all conditions:
#   SEED=0 EPOCHS=10 bash scripts_Exp2_dir2/run_all_ablation_dir2.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=https://hf-mirror.com

START=$(date +%s)
echo "=============================================="
echo " Exp-2 Dir2 (Narrative/Manipulation) Leave-One-Out Ablation — 8 Conditions"
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

run_condition "scripts_Exp2_dir2/run_dir2_full.sh"                          "dir2_full                        (all 7 fields)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_narrative_frame.sh"            "dir2_wo_narrative_frame           (- narrative_frame)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_persuasion_cues.sh"            "dir2_wo_persuasion_cues           (- persuasion_cues)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_engagement_pattern.sh"         "dir2_wo_engagement_pattern        (- engagement_pattern)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_coordination_signals.sh"       "dir2_wo_coordination_signals      (- coordination_signals)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_evidence_visibility.sh"        "dir2_wo_evidence_visibility       (- evidence_visibility)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_attention_triggers.sh"         "dir2_wo_attention_triggers        (- attention_triggers)"
run_condition "scripts_Exp2_dir2/run_dir2_wo_manipulation_risk_profile.sh"  "dir2_wo_manipulation_risk_profile (- manipulation_risk_profile)"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "=============================================="
echo " All 8 conditions complete."
echo " Total time: $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "=============================================="
echo ""
echo "Next: python scripts_Exp2_dir2/eval_ablation_dir2.py"
