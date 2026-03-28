#!/usr/bin/env bash
# run_all_models.sh — Run the full lab pipeline using existing scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${ENV_NAME:-env_214}"
SKIP_FEATURE_PIPELINE="${SKIP_FEATURE_PIPELINE:-1}"

# Allow positional override: bash run_all_models.sh 1
if [[ $# -ge 1 ]]; then
    SKIP_FEATURE_PIPELINE="$1"
fi

# Activate conda environment once for the full pipeline.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "========================================"
echo "Full pipeline start"
echo "Project root: $PROJECT_ROOT"
echo "Conda env:    $ENV_NAME"
echo "Skip stage 1: $SKIP_FEATURE_PIPELINE"
echo "========================================"

run_step() {
    local label="$1"
    local script_path="$2"

    echo ""
    echo "----------------------------------------"
    echo "$label"
    echo "Script: $script_path"
    echo "----------------------------------------"

    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Missing script $script_path" >&2
        exit 1
    fi

    bash "$script_path"
}

if [[ "$SKIP_FEATURE_PIPELINE" == "1" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "1) Autoencoder + feature pipeline"
    echo "Skipped (SKIP_FEATURE_PIPELINE=1)"
    echo "----------------------------------------"
else
    run_step "1) Autoencoder + feature pipeline" "$SCRIPT_DIR/run.sh"
fi
run_step "2) Logistic Regression + SVM" "$SCRIPT_DIR/run_logreg_svm.sh"
run_step "3) Ensemble training (RF + HGB)" "$SCRIPT_DIR/run_ensemble_train.sh"
run_step "4) Ensemble analysis" "$SCRIPT_DIR/run_ensemble_analysis.sh"

echo ""
echo "========================================"
echo "Full pipeline completed successfully."
echo "========================================"
