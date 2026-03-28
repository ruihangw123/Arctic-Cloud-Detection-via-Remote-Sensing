#!/usr/bin/env bash
# run_ensemble_train.sh — Run ensemble.py model script

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

echo "=== Ensemble Model Script ==="
echo "Project root: $PROJECT_ROOT"

# Verify required input data
TRAIN_DATA="$PROJECT_ROOT/feature_eng_dataset/train_features_opt.csv"
TEST_DATA="$PROJECT_ROOT/feature_eng_dataset/test_features_opt.csv"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: Training data not found at $TRAIN_DATA" >&2
    exit 1
fi
if [[ ! -f "$TEST_DATA" ]]; then
    echo "ERROR: Test data not found at $TEST_DATA" >&2
    exit 1
fi

echo "Training data:  $TRAIN_DATA"
echo "Test data:      $TEST_DATA"
echo ""

# Run ensemble script
"$PYTHON_BIN" "$PROJECT_ROOT/code/models/ensemble.py"

echo ""
echo "=== Done. Results saved to $PROJECT_ROOT/results/ ==="
