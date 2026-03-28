#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"
CKPT="../results/checkpoints/ae_opt-best.ckpt"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither python nor python3 is available in PATH." >&2
    exit 127
fi

echo "========================================"
echo "Train autoencoder"
echo "========================================"
"$PYTHON_BIN" run_autoencoder.py "$CONFIG"

echo "========================================"
echo "Extract embeddings for labeled images"
echo "========================================"
"$PYTHON_BIN" get_embedding.py "$CONFIG" "$CKPT"

echo "========================================"
echo "Build final train/test datasets"
echo "========================================"
"$PYTHON_BIN" feature_engineering_autoencoder.py "$CONFIG"

echo "========================================"
echo "Datasets checks"
echo "========================================"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd

df_train = pd.read_csv("../feature_eng_dataset/train_features_opt.csv")
df_test = pd.read_csv("../feature_eng_dataset/test_features_opt.csv")

ae_cols = [c for c in df_train.columns if c.startswith("ae")]
if len(ae_cols) == 0:
    raise ValueError("No AE columns found in train_features_opt.csv")

print("\nTrain shape:", df_train.shape)
print("Test shape :", df_test.shape)
print("AE dim     :", len(ae_cols))
print("Train null :", int(df_train.isnull().sum().sum()))
print("Test null  :", int(df_test.isnull().sum().sum()))

print("\nTrain image counts:")
print(df_train["image"].value_counts(dropna=False))

print("\nTest image counts:")
print(df_test["image"].value_counts(dropna=False))
PY

echo "========================================"
echo "Pipeline finished successfully."
echo "========================================"
