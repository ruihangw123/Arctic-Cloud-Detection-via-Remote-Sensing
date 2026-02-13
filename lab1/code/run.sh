#!/bin/bash
# Run Lab 1 analysis to reproduce results and generate figures.
# Execute from project root: bash code/run.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Lab 1: PECARN TBI Analysis ==="

run_python() {
    if command -v conda &>/dev/null; then
        conda run -n stat214 python "$@" || python "$@"
    else
        python "$@"
    fi
}

echo "1. Running data cleaning and exploratory analysis..."
run_python code/data_analysis.py

echo ""
echo "2. Running models and generating figures..."
run_python code/run_analysis.py

echo "=== Done ==="
