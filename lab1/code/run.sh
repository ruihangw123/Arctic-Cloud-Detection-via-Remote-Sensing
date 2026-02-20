#!/bin/bash
# Run Lab 1 analysis: clean data, fit models, generate figures.
# Execute from the lab1 directory: bash code/run.sh

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
