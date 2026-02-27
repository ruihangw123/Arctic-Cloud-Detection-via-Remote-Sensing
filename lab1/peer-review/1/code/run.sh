#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stat214
python clean.py
python models.py
echo "Done."
conda deactivate
