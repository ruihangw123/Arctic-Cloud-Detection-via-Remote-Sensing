#!/usr/bin/env bash
# run_logreg_svm.sh — Run logistic regression and SVM models + stability analysis

set -euo pipefail

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_214

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither python nor python3 is available in PATH." >&2
    exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Logistic Regression & SVM ==="
echo "Project root: $PROJECT_ROOT"

# Verify required input data
TRAIN_DATA="$PROJECT_ROOT/feature_eng_dataset/train_features_opt.csv"
TEST_DATA="$PROJECT_ROOT/feature_eng_dataset/test_features_opt.csv"

for f in "$TRAIN_DATA" "$TEST_DATA"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required data not found at $f" >&2
        exit 1
    fi
done

echo "Train data: $TRAIN_DATA"
echo "Test data:  $TEST_DATA"
echo ""

# Run main models
echo "1) Running logreg + SVM ..."
"$PYTHON_BIN" "$PROJECT_ROOT/code/models/logreg_svm.py"

# Run stability analysis
echo ""
echo "2) Running label-flip stability ..."
"$PYTHON_BIN" "$PROJECT_ROOT/code/models/logreg_svm_stability.py"

echo ""
echo "=== Done. ==="
