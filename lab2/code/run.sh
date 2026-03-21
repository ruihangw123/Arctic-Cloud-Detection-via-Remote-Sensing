#!/bin/bash
#SBATCH --job-name=stat214-lab2
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
SKIP_AE=false
AE_MODE="train_local"
# AE_CONFIG="configs/local.yaml"
AE_CONFIG="configs/autoencoder.yaml"
AE_N_IMAGES=30
AE_MAX_PATCHES=300000
USE_WANDB=false
SEED=42

for arg in "$@"; do
    case "$arg" in
        --skip-autoencoder) SKIP_AE=true ;;
        --ae-mode=*)        AE_MODE="${arg#*=}" ;;
        --ae-config=*)      AE_CONFIG="${arg#*=}" ;;
        --ae-n-images=*)    AE_N_IMAGES="${arg#*=}" ;;
        --wandb)            USE_WANDB=true ;;
        --seed=*)           SEED="${arg#*=}" ;;
    esac
done

# ── Working directory ────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

DATA="$PWD/../data"
REPORT="$PWD/../report/figures"
PREDICT="$PWD/../data/predictions"
LOG="$PWD/../report/full_modeling_run.log"
AE_EMBED_DIR="$DATA"

mkdir -p "$REPORT" "$PREDICT" "$PWD/../report"

echo "===================================================="
echo "  STAT 214 Lab 2 — Full Pipeline"
echo "  Node:       $(hostname) | $(date)"
echo "  Data dir:   $DATA"
echo "  AE mode:    $AE_MODE  (skip=$SKIP_AE)"
echo "  AE config:  $AE_CONFIG"
echo "  Seed:       $SEED"
echo "===================================================="

# ── Conda environment ────────────────────────────────────────────────────────
# for f in /usr/local/linux/miniforge-*/etc/profile.d/conda.sh \
#          "$HOME"/miniconda3/etc/profile.d/conda.sh \
#          "$HOME"/miniforge3/etc/profile.d/conda.sh; do
#     [ -f "$f" ] && source "$f" && break
# done

# if ! conda env list 2>/dev/null | grep -q '^env_214\s'; then
#     echo ">>> Creating conda env env_214..."
#     mamba env create -y -f env_scf.yaml 2>/dev/null || conda env create -y -f env_scf.yaml
# else
#     echo ">>> Updating conda env env_214..."
#     mamba env update -f env_scf.yaml --prune 2>/dev/null || conda env update -f env_scf.yaml --prune
# fi
# echo ">>> Activating env_214"
# conda activate env_214

module load anaconda3
conda activate env_214

# Sanity check
N_FILES=$(find "$DATA" -maxdepth 1 -name '*.npz' | wc -l)
[ "$N_FILES" -eq 0 ] && { echo "ERROR: No .npz files in $DATA"; exit 1; }
echo ">>> Found $N_FILES .npz files"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Train autoencoder (unsupervised, on all images)
# ═════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_AE" = false ]; then
    echo ""
    echo ">>> Step 1a: Training autoencoder (mode=$AE_MODE)..."

    WANDB_FLAG=""
    [ "$USE_WANDB" = true ] && WANDB_FLAG="--use-wandb"

    python autoencoder.py \
        --mode "$AE_MODE" \
        --config "$AE_CONFIG" \
        --n-images "$AE_N_IMAGES" \
        --max-patches "$AE_MAX_PATCHES" \
        --seed "$SEED" \
        $WANDB_FLAG
        

    # ── Step 1b: Export embeddings from best checkpoint ──────────────────────
    echo ""
    echo ">>> Step 1b: Exporting autoencoder embeddings..."

    if [ "$AE_MODE" = "train_full" ]; then
        CKPT_PATTERN="checkpoints/full-*.ckpt"
    else
        # CKPT_PATTERN="checkpoints/local-*.ckpt"
        CKPT_PATTERN="checkpoints/model-*.ckpt"
    fi

    # Sort by val_loss (ascending) extracted from filename, pick the best
    BEST_CKPT=$(ls $CKPT_PATTERN 2>/dev/null | sed 's/.*val_loss=//' | paste -d' ' - <(ls $CKPT_PATTERN 2>/dev/null) | sort -n | head -1 | awk '{print $2}')
    [ -z "$BEST_CKPT" ] && BEST_CKPT=$(ls -t $CKPT_PATTERN 2>/dev/null | head -1 || true)
    if [ -z "$BEST_CKPT" ]; then
        echo "  WARNING: No checkpoint found ($CKPT_PATTERN). Skipping embedding export."
        echo "  Classification will run without AE features."
        AE_EMBED_DIR=""
    else
        echo "  Using checkpoint: $BEST_CKPT"
        python autoencoder.py \
            --mode embed \
            --config "$AE_CONFIG" \
            --checkpoint "$BEST_CKPT" \
            --output-dir "$AE_EMBED_DIR"
        echo "  Embeddings exported to: $AE_EMBED_DIR"
    fi
else
    echo ""
    echo ">>> Skipping autoencoder (--skip-autoencoder)"
    # Check if embeddings already exist from a previous run
    AE_COUNT=$(find "$AE_EMBED_DIR" -maxdepth 1 -name '*_ae.csv' 2>/dev/null | wc -l)
    if [ "$AE_COUNT" -gt 0 ]; then
        echo "  Found $AE_COUNT existing *_ae.csv files — will use them."
    else
        echo "  No existing AE embeddings found — running without AE features."
        AE_EMBED_DIR=""
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Classification with 3-fold CV + 4 models + visualizations
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Running classification pipeline (4 models × 3 CV folds)..."

AE_FLAG=""
[ -n "$AE_EMBED_DIR" ] && AE_FLAG="--ae-embed-dir $AE_EMBED_DIR"

python modeling.py \
    --mode advanced \
    --n-files 3 \
    --max-samples-per-image 0 \
    --feature-mode enhanced \
    --models logreg rf gbm knn \
    --run-ablation \
    --output-dir "$REPORT" \
    --predictions-dir "$PREDICT" \
    --data-dir "$DATA" \
    --log-file "$LOG" \
    --seed "$SEED" \
    $AE_FLAG

# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Stability Analysis (PCS framework)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: Running stability analysis..."
python stability.py \
    --data-dir "$DATA" \
    --output-dir "$REPORT" \
    --n-bootstrap 30 \
    --model logreg \
    --feature-mode enhanced \
    $AE_FLAG \
    --seed "$SEED"

# ═════════════════════════════════════════════════════════════════════════════
# Done
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "===================================================="
echo "  Pipeline complete!  $(date)"
echo "  Figures:      $REPORT/"
echo "  Predictions:  $PREDICT/"
echo "  Log:          $LOG"
echo "===================================================="
echo ""
echo "Generated visualizations:"
ls -1 "$REPORT"/*.png 2>/dev/null || echo "  (none found)"
echo ""
echo "Prediction files:"
find "$PREDICT" -name '*.csv' 2>/dev/null | wc -l
echo " CSV files in $PREDICT/"
