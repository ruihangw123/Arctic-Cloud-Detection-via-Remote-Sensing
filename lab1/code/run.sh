#!/bin/bash
# Reproduce all Lab 1 results and figures.
# Usage: bash code/run.sh

set -e
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate stat214

echo "=== Lab 1: PECARN TBI Analysis ==="

echo "1. Running data cleaning..."
python code/clean.py

echo ""
echo "2. Running models and generating figures..."
MPLBACKEND=Agg python code/models.py

echo ""
echo "=== Done ==="

conda deactivate
