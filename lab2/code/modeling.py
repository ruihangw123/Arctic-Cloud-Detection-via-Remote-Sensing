import argparse
import glob
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from data import make_labeled_table_low_ram


EPS = 1e-6
RAW_FEATURE_NAMES = ["NDAI", "SD", "CORR", "Rad_DF", "Rad_CF", "Rad_BF", "Rad_AF", "Rad_AN"]
RADIANCE_NAMES = ["Rad_DF", "Rad_CF", "Rad_BF", "Rad_AF", "Rad_AN"]


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    test_coords: np.ndarray = None
    test_image_name: str = ""
    test_all_X: np.ndarray = None      # features for ALL pixels (incl. unlabeled)
    test_all_coords: np.ndarray = None  # coords  for ALL pixels (incl. unlabeled)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


## Data helpers

def resolve_data_dir(data_dir_arg):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []
    if os.path.isabs(data_dir_arg):
        candidates.append(os.path.normpath(data_dir_arg))
    else:
        candidates.append(os.path.normpath(os.path.join(os.getcwd(), data_dir_arg)))
        candidates.append(os.path.normpath(os.path.join(script_dir, data_dir_arg)))

    candidates.append(os.path.normpath(os.path.join(script_dir, "../data")))

    seen = set()
    ordered = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)

    for path in ordered:
        if os.path.isdir(path):
            return path, ordered

    return ordered[0], ordered


def load_npz_array(file_path):
    npz_data = np.load(file_path)
    key = list(npz_data.files)[0]
    return npz_data[key]


def discover_labeled_files(data_dir):
    labeled = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        arr = load_npz_array(fp)
        if arr.shape[1] == 11:
            labeled.append(fp)
    return labeled


## Autoencoder embedding helpers

def load_ae_embeddings(ae_embed_dir, image_basename):
    """Load AE embedding CSV for a given image and return (coords, features, names).

    Returns None if the file does not exist.
    """
    stem = os.path.splitext(image_basename)[0]
    csv_path = os.path.join(ae_embed_dir, f"{stem}_ae.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    ae_cols = [c for c in df.columns if c.startswith("ae")]
    coords = df[["y", "x"]].values.astype(int)
    features = df[ae_cols].values.astype(np.float32)
    return coords, features, ae_cols


def merge_ae_features(raw_data, ae_embed_dir, image_basename):
    """Join AE embeddings to raw pixel data by (y, x).

    Returns the AE feature columns aligned to raw_data rows, or None if
    embeddings are unavailable.
    """
    result = load_ae_embeddings(ae_embed_dir, image_basename)
    if result is None:
        return None, []
    ae_coords, ae_feats, ae_names = result

    pixel_coords = raw_data[:, :2].astype(int)
    coord_to_idx = {}
    for i, (cy, cx) in enumerate(ae_coords):
        coord_to_idx[(cy, cx)] = i

    n = len(pixel_coords)
    n_ae = ae_feats.shape[1]
    out = np.zeros((n, n_ae), dtype=np.float32)
    matched = 0
    for j in range(n):
        key = (pixel_coords[j, 0], pixel_coords[j, 1])
        if key in coord_to_idx:
            out[j] = ae_feats[coord_to_idx[key]]
            matched += 1

    if matched < n * 0.5:
        print(f"  WARNING: AE merge matched only {matched}/{n} pixels for {image_basename}")
    return out, list(ae_names)


## Feature engineering

def compute_spatial_features(arr):
    """Compute spatial neighbourhood features on the full image array (before label filtering).

    Must be called on the complete image so that neighbourhood lookups are valid.
    Returns (lookup, names) where lookup maps (y, x) -> feature row (np.float32).
    """
    try:
        from feature_engineering import rank_difference, pooled_rank_difference, pooled_feature, sobel_gradient
    except ImportError:
        return {}, []

    cols = ["y", "x", "NDAI", "SD", "CORR", "Rad_DF", "Rad_CF", "Rad_BF", "Rad_AF", "Rad_AN"]
    n_cols = min(arr.shape[1], 10)
    df = pd.DataFrame(arr[:, :n_cols], columns=cols[:n_cols])

    # Multi-scale patch features: 3×3, 9×9, 15×15
    spatial_names = [
        "rank_diff", "pooled_rank_diff",
        # 3x3
        "NDAI_mean_p3", "NDAI_sd_p3", "CORR_mean_p3",
        # 9x9
        "NDAI_mean_p9", "NDAI_sd_p9", "CORR_mean_p9", "SD_mean_p9",
        # 15x15
        "NDAI_mean_p15", "NDAI_sd_p15", "CORR_mean_p15",
        # gradients
        "sobel_NDAI", "sobel_CORR",
    ]
    feat_matrix = np.column_stack([
        rank_difference(df),
        pooled_rank_difference(df),
        # 3x3
        pooled_feature(df, "NDAI", "mean", patch_size=3),
        pooled_feature(df, "NDAI", "sd",   patch_size=3),
        pooled_feature(df, "CORR", "mean", patch_size=3),
        # 9x9
        pooled_feature(df, "NDAI", "mean", patch_size=9),
        pooled_feature(df, "NDAI", "sd",   patch_size=9),
        pooled_feature(df, "CORR", "mean", patch_size=9),
        pooled_feature(df, "SD",   "mean", patch_size=9),
        # 15x15
        pooled_feature(df, "NDAI", "mean", patch_size=15),
        pooled_feature(df, "NDAI", "sd",   patch_size=15),
        pooled_feature(df, "CORR", "mean", patch_size=15),
        # gradients
        sobel_gradient(df, "NDAI"),
        sobel_gradient(df, "CORR"),
    ]).astype(np.float32)

    keys = list(zip(arr[:, 0].astype(int), arr[:, 1].astype(int)))
    return dict(zip(keys, feat_matrix)), spatial_names


def build_features(raw_data, mode="enhanced", ae_embed_dir=None, image_basename=None,
                   spatial_lookup=None, spatial_names=None):
    """
    raw_data columns:
      [y, x, NDAI, SD, CORR, Rad_DF, Rad_CF, Rad_BF, Rad_AF, Rad_AN, label]
    """
    base = raw_data[:, 2:10].astype(np.float32)

    if mode == "expert":
        return base[:, :3], ["NDAI", "SD", "CORR"]

    if mode == "radiance":
        return base[:, 3:8], RADIANCE_NAMES.copy()

    if mode == "raw_all":
        return base, RAW_FEATURE_NAMES.copy()

    if mode not in ("enhanced", "enhanced_no_ae", "enhanced_no_spatial"):
        raise ValueError(f"Unknown feature mode: {mode}")

    ndai, sd, corr = base[:, 0:1], base[:, 1:2], base[:, 2:3]
    rad = base[:, 3:8]

    sd_log = np.log1p(np.clip(sd, 0.0, None))
    rad_log = np.log1p(np.clip(rad, 0.0, None))

    # Angular anisotropy: normalised difference vs nadir-like AN angle
    an = rad[:, 4:5]
    aniso = (rad[:, :4] - an) / (rad[:, :4] + an + EPS)

    adj_ratio = np.concatenate([
        rad[:, 0:1] / (rad[:, 1:2] + EPS),
        rad[:, 1:2] / (rad[:, 2:3] + EPS),
        rad[:, 2:3] / (rad[:, 3:4] + EPS),
        rad[:, 3:4] / (rad[:, 4:5] + EPS),
    ], axis=1)

    interactions = np.concatenate([ndai * corr, ndai * sd_log, corr * sd_log], axis=1)

    X = np.concatenate([base, sd_log, rad_log, aniso, adj_ratio, interactions], axis=1)
    names = (
        RAW_FEATURE_NAMES
        + ["SD_log"]
        + [f"log_{n}" for n in RADIANCE_NAMES]
        + ["aniso_DF_AN", "aniso_CF_AN", "aniso_BF_AN", "aniso_AF_AN"]
        + ["ratio_DF_CF", "ratio_CF_BF", "ratio_BF_AF", "ratio_AF_AN"]
        + ["NDAI_x_CORR", "NDAI_x_SDlog", "CORR_x_SDlog"]
    )

    if ae_embed_dir and image_basename and mode != "enhanced_no_ae":
        ae_feats, ae_names = merge_ae_features(raw_data, ae_embed_dir, image_basename)
        if ae_feats is not None:
            X = np.concatenate([X, ae_feats], axis=1)
            names = names + ae_names

    if spatial_lookup and spatial_names and mode != "enhanced_no_spatial":
        n_sp = len(spatial_names)
        sp_out = np.zeros((len(raw_data), n_sp), dtype=np.float32)
        for j, (cy, cx) in enumerate(raw_data[:, :2].astype(int)):
            row = spatial_lookup.get((cy, cx))
            if row is not None:
                sp_out[j] = row
        X = np.concatenate([X, sp_out], axis=1)
        names = names + spatial_names

    return X.astype(np.float32), names


def sample_labeled_rows(raw_data, max_samples_per_image, rng):
    labels = raw_data[:, 10].astype(int)
    mask = labels != 0
    labeled = raw_data[mask]

    if max_samples_per_image is not None and len(labeled) > max_samples_per_image:
        idx = rng.choice(len(labeled), size=max_samples_per_image, replace=False)
        labeled = labeled[idx]
    return labeled


## Train / val / test split (image-level)

def get_image_level_split(data_dir, n_files, max_samples_per_image, feature_mode, seed,
                          ae_embed_dir=None):
    rng = np.random.default_rng(seed)
    resolved_data_dir, checked_paths = resolve_data_dir(data_dir)
    labeled_files = discover_labeled_files(resolved_data_dir)

    if len(labeled_files) == 0:
        checked_msg = "\n  - " + "\n  - ".join(checked_paths)
        raise ValueError(
            "No labeled .npz files found (expected arrays with 11 columns).\n"
            f"Checked data directories:{checked_msg}\n"
            "Tip: pass --data-dir explicitly, e.g. --data-dir ../data or --data-dir ./data"
        )

    n_files = min(n_files, len(labeled_files))
    selected = labeled_files[:n_files]

    print(f"Resolved data directory: {resolved_data_dir}")
    print("Selected labeled files:")
    for fp in selected:
        print(f"  - {os.path.basename(fp)}")

    image_blocks = []
    for fp in tqdm(selected, desc="Loading selected files", leave=False):
        arr = load_npz_array(fp)
        spatial_lookup, spatial_names = compute_spatial_features(arr)
        raw_labeled = sample_labeled_rows(arr, max_samples_per_image=max_samples_per_image, rng=rng)
        X_img, names = build_features(
            raw_labeled, mode=feature_mode,
            ae_embed_dir=ae_embed_dir, image_basename=os.path.basename(fp),
            spatial_lookup=spatial_lookup, spatial_names=spatial_names,
        )
        y_img = (raw_labeled[:, 10].astype(int) == 1).astype(np.int64)
        coords_img = raw_labeled[:, :2]
        image_blocks.append((os.path.basename(fp), X_img, y_img, coords_img))

    test_coords = None
    test_image_name = ""

    if len(image_blocks) >= 4:
        train_blocks = image_blocks[:-2]
        val_block = image_blocks[-2]
        test_block = image_blocks[-1]

        X_train = np.concatenate([b[1] for b in train_blocks])
        y_train = np.concatenate([b[2] for b in train_blocks])
        X_val, y_val = val_block[1], val_block[2]
        X_test, y_test = test_block[1], test_block[2]
        test_coords = test_block[3]
        test_image_name = test_block[0]

        print("\nSplit strategy (image-level, leakage-safe):")
        print(f"  Train images: {[b[0] for b in train_blocks]}")
        print(f"  Val image:    {val_block[0]}")
        print(f"  Test image:   {test_block[0]}")

    elif len(image_blocks) == 3:
        train_blocks = image_blocks[:2]
        test_block = image_blocks[2]

        X_trainval = np.concatenate([b[1] for b in train_blocks])
        y_trainval = np.concatenate([b[2] for b in train_blocks])

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=0.2, random_state=seed, stratify=y_trainval,
        )
        X_test, y_test = test_block[1], test_block[2]
        test_coords = test_block[3]
        test_image_name = test_block[0]

        print("\nSplit strategy (recommended for 3 labeled files):")
        print(f"  Train/Val images: {[b[0] for b in train_blocks]} (stratified 80/20 split)")
        print(f"  Test image:       {test_block[0]} (fully unseen)")

    elif len(image_blocks) == 2:
        train_block = image_blocks[0]
        test_block = image_blocks[1]

        X_train_full, y_train_full = train_block[1], train_block[2]
        n_val = max(1, int(0.2 * len(X_train_full)))
        perm = rng.permutation(len(X_train_full))

        X_train, y_train = X_train_full[perm[n_val:]], y_train_full[perm[n_val:]]
        X_val, y_val = X_train_full[perm[:n_val]], y_train_full[perm[:n_val]]
        X_test, y_test = test_block[1], test_block[2]
        test_coords = test_block[3]
        test_image_name = test_block[0]

        print("\nSplit strategy:")
        print(f"  Train image:  {train_block[0]} (with in-image val split)")
        print(f"  Test image:   {test_block[0]} (fully unseen)")

    else:
        X_all, y_all = image_blocks[0][1], image_blocks[0][2]
        coords_all = image_blocks[0][3]
        perm = rng.permutation(len(X_all))
        n_train = int(0.6 * len(X_all))
        n_val = int(0.2 * len(X_all))

        X_train = X_all[perm[:n_train]]
        y_train = y_all[perm[:n_train]]
        X_val = X_all[perm[n_train:n_train + n_val]]
        y_val = y_all[perm[n_train:n_train + n_val]]
        X_test = X_all[perm[n_train + n_val:]]
        y_test = y_all[perm[n_train + n_val:]]
        test_coords = coords_all[perm[n_train + n_val:]]
        test_image_name = image_blocks[0][0]

        print("\nWarning: only 1 labeled image. Pixel-level split may leak spatial info.")

    return DatasetSplit(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=names,
        test_coords=test_coords,
        test_image_name=test_image_name,
    )


## 3-Fold image-level cross-validation

def get_cross_validation_splits(data_dir, max_samples_per_image, feature_mode, seed,
                                ae_embed_dir=None):
    """Generate 3-fold CV splits rotating labeled images through train/val/test.

    Each fold uses one image for training (80 % fit, 20 % held out),
    one image as the validation set (threshold tuning), and one as the
    fully-unseen test set.
    """
    rng = np.random.default_rng(seed)
    resolved_data_dir, checked_paths = resolve_data_dir(data_dir)
    labeled_files = discover_labeled_files(resolved_data_dir)

    if len(labeled_files) < 3:
        checked_msg = "\n  - " + "\n  - ".join(checked_paths)
        raise ValueError(
            f"Need at least 3 labeled images for cross-validation, found {len(labeled_files)}.\n"
            f"Checked:{checked_msg}"
        )

    selected = labeled_files[:3]
    print(f"Resolved data directory: {resolved_data_dir}")
    print("Labeled images for cross-validation:")
    for fp in selected:
        print(f"  - {os.path.basename(fp)}")

    image_blocks = []
    raw_arrays = []  # keep full arrays for all-pixel prediction
    spatial_lookups = []
    for fp in tqdm(selected, desc="Loading labeled images", leave=False):
        arr = load_npz_array(fp)
        spatial_lookup, spatial_names = compute_spatial_features(arr)
        raw_labeled = sample_labeled_rows(arr, max_samples_per_image=max_samples_per_image, rng=rng)
        X_img, names = build_features(
            raw_labeled, mode=feature_mode,
            ae_embed_dir=ae_embed_dir, image_basename=os.path.basename(fp),
            spatial_lookup=spatial_lookup, spatial_names=spatial_names,
        )
        y_img = (raw_labeled[:, 10].astype(int) == 1).astype(np.int64)
        coords_img = raw_labeled[:, :2]
        image_blocks.append((os.path.basename(fp), X_img, y_img, coords_img))
        raw_arrays.append(arr)
        spatial_lookups.append((spatial_lookup, spatial_names))

    folds = []
    for fold_idx in range(3):
        # Leave-one-out: 2 images train, 1 image test
        test_idx = fold_idx
        train_idxs = [(fold_idx + 1) % 3, (fold_idx + 2) % 3]

        test_block = image_blocks[test_idx]

        # Merge two training images
        X_tr = np.concatenate([image_blocks[i][1] for i in train_idxs])
        y_tr = np.concatenate([image_blocks[i][2] for i in train_idxs])

        # Hold out 20% of training data as validation (for threshold tuning)
        n_val = int(len(X_tr) * 0.2)
        perm = rng.permutation(len(X_tr))
        val_mask = perm[:n_val]
        train_mask = perm[n_val:]

        X_val, y_val = X_tr[val_mask], y_tr[val_mask]
        X_train_final, y_train_final = X_tr[train_mask], y_tr[train_mask]

        train_names = [image_blocks[i][0] for i in train_idxs]

        # Build features for ALL pixels of test image (incl. unlabeled)
        test_arr = raw_arrays[test_idx]
        test_sl, test_sn = spatial_lookups[test_idx]
        X_all, _ = build_features(
            test_arr, mode=feature_mode,
            ae_embed_dir=ae_embed_dir,
            image_basename=os.path.basename(selected[test_idx]),
            spatial_lookup=test_sl, spatial_names=test_sn,
        )
        coords_all = test_arr[:, :2]

        split = DatasetSplit(
            X_train=X_train_final, y_train=y_train_final,
            X_val=X_val, y_val=y_val,
            X_test=test_block[1], y_test=test_block[2],
            feature_names=names,
            test_coords=test_block[3],
            test_image_name=test_block[0],
            test_all_X=X_all,
            test_all_coords=coords_all,
        )

        folds.append({
            "fold_idx": fold_idx,
            "train_image": "+".join(train_names),
            "val_image": "(20% holdout)",
            "test_image": test_block[0],
            "split": split,
        })

        print(f"  Fold {fold_idx + 1}: Train={'+'.join(train_names)} ({len(X_train_final)} samples), "
              f"Val=({n_val} holdout), Test={test_block[0]} ({len(X_all)} total pixels)")

    return folds, names


## Winsorization

def fit_winsor_bounds(X_train, q_low=0.01, q_high=0.99):
    low = np.quantile(X_train, q_low, axis=0)
    high = np.quantile(X_train, q_high, axis=0)
    return low, high


def apply_winsor(X, low, high):
    n = X.shape[1]
    return np.clip(X, low[:n], high[:n])


## Threshold & metrics

def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + EPS)
    return float(thresholds[int(np.argmax(f1))])


def find_thresholds_multi_strategy(y_true, y_prob, min_recall=0.90, min_precision=0.90):
    """Return thresholds under three strategies:
      - 'f1':          maximise F1
      - 'prec@rec90':  highest precision subject to recall >= min_recall
      - 'rec@prec90':  highest recall subject to precision >= min_precision
    Returns dict mapping strategy name -> threshold (float).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + EPS)

    best_f1_thr = float(thresholds[np.argmax(f1)]) if len(thresholds) else 0.5

    # prec@rec90
    mask_rec = recall[:-1] >= min_recall
    if mask_rec.any():
        idx = np.argmax(precision[:-1][mask_rec])
        prec_thr = float(thresholds[np.where(mask_rec)[0][idx]])
    else:
        prec_thr = best_f1_thr

    # rec@prec90
    mask_prec = precision[:-1] >= min_precision
    if mask_prec.any():
        idx = np.argmax(recall[:-1][mask_prec])
        rec_thr = float(thresholds[np.where(mask_prec)[0][idx]])
    else:
        rec_thr = best_f1_thr

    return {"f1": best_f1_thr, "prec@rec90": prec_thr, "rec@prec90": rec_thr}


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
        "pred_pos_rate": float(y_pred.mean()),
        "true_pos_rate": float(y_true.mean()),
    }


## Model building & evaluation

def build_model(model_name, seed):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed,
        )
    if model_name == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5, random_state=seed,
        )
    if model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
        ])
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model_name, split, seed):
    """Train model and return result dict with model, probabilities, and metrics."""
    model = build_model(model_name, seed=seed)
    model.fit(split.X_train, split.y_train)

    val_prob = model.predict_proba(split.X_val)[:, 1]
    test_prob = model.predict_proba(split.X_test)[:, 1]

    best_thr = find_best_threshold(split.y_val, val_prob)
    val_metrics = compute_metrics(split.y_val, val_prob, threshold=best_thr)
    test_metrics = compute_metrics(split.y_test, test_prob, threshold=best_thr)

    return {
        "model_name": model_name,
        "model": model,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold": best_thr,
    }


def print_metrics_table(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"acc={metrics['accuracy']:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} | "
        f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f}"
    )
    print(
        f"roc_auc={metrics['roc_auc']:.4f} | pr_auc={metrics['pr_auc']:.4f} | "
        f"thr={metrics['threshold']:.4f}"
    )
    print(
        f"confusion: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']} | "
        f"pred_pos_rate={metrics['pred_pos_rate']:.3f}, true_pos_rate={metrics['true_pos_rate']:.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization functions
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    'logreg': '#2176AE',
    'rf': '#F57C20',
    'gbm': '#3A9B5B',
    'knn': '#D63B3B',
}
MODEL_LABELS = {
    'logreg': 'Logistic Regression',
    'rf': 'Random Forest',
    'gbm': 'Gradient Boosting',
    'knn': 'KNN',
}


def setup_plot_style():
    """Set up a clean, publication-quality matplotlib style."""
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def _save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(results_list, split, output_dir):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for res in results_list:
        fpr, tpr, _ = roc_curve(split.y_test, res["test_prob"])
        auc_val = res["test_metrics"]["roc_auc"]
        color = MODEL_COLORS.get(res["model_name"], None)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS.get(res['model_name'], res['model_name'])} (AUC={auc_val:.4f})",
                linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir, "roc_curves.png")


def plot_pr_curves(results_list, split, output_dir):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for res in results_list:
        prec, rec, _ = precision_recall_curve(split.y_test, res["test_prob"])
        ap = res["test_metrics"]["pr_auc"]
        color = MODEL_COLORS.get(res["model_name"], None)
        ax.plot(rec, prec, label=f"{MODEL_LABELS.get(res['model_name'], res['model_name'])} (AP={ap:.4f})",
                linewidth=2, color=color)
    baseline = split.y_test.mean()
    ax.axhline(baseline, color="gray", ls="--", alpha=0.5, label=f"baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — Test Set")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir, "pr_curves.png")


def plot_confusion_matrices(results_list, split, output_dir):
    setup_plot_style()
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results_list):
        y_pred = (res["test_prob"] >= res["threshold"]).astype(int)
        cm = confusion_matrix(split.y_test, y_pred)
        total = cm.sum()
        # Create annotations with counts and percentages
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm[i, j] / total:.1%})"
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                    xticklabels=["No Cloud", "Cloud"], yticklabels=["No Cloud", "Cloud"],
                    ax=ax)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(f"{MODEL_LABELS.get(res['model_name'], res['model_name'])} (thr={res['threshold']:.3f})")
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir, "confusion_matrices.png")


def plot_feature_importance(results_list, split, output_dir):
    setup_plot_style()
    for res in results_list:
        model = res["model"]
        name = res["model_name"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", None)
            if clf is not None and hasattr(clf, "coef_"):
                imp = np.abs(clf.coef_[0])
            else:
                continue
        else:
            continue

        feat_names = split.feature_names if hasattr(split, 'feature_names') else [f"f{i}" for i in range(len(imp))]
        idx = np.argsort(imp)[::-1][:20]

        fig, ax = plt.subplots(figsize=(7, 6))
        y_pos = np.arange(len(idx))
        c = MODEL_COLORS.get(name, '#2176AE')

        # Lollipop chart
        ax.hlines(y=y_pos, xmin=0, xmax=imp[idx], color=c, alpha=0.7, linewidth=1.5)
        ax.scatter(imp[idx], y_pos, color=c, s=40, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat_names[i] for i in idx])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"{MODEL_LABELS.get(name, name)} — Top 20 Features")
        fig.tight_layout()
        _save_fig(fig, output_dir, f"feature_importance_{name}.png")

    # Combined plot: RF/GBM vs LogReg
    tree_models = [r for r in results_list if r["model_name"] in ("rf", "gbm")]
    linear_models = [r for r in results_list if r["model_name"] == "logreg"]

    if tree_models and linear_models:
        rf_or_gbm = tree_models[0]
        lr = linear_models[0]

        imp_tree = rf_or_gbm["model"].feature_importances_
        lr_model = lr["model"]
        if hasattr(lr_model, "named_steps"):
            imp_lr = np.abs(lr_model.named_steps["clf"].coef_[0])
        else:
            imp_lr = np.abs(lr_model.coef_[0])
        feat_names = split.feature_names if hasattr(split, 'feature_names') else [f"f{i}" for i in range(len(imp_tree))]

        # Union of top 15 from each
        top_tree = set(np.argsort(imp_tree)[::-1][:15])
        top_lr = set(np.argsort(imp_lr)[::-1][:15])
        union_idx = sorted(top_tree | top_lr, key=lambda i: imp_tree[i], reverse=True)[:20]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
        y_pos = np.arange(len(union_idx))

        c1 = MODEL_COLORS[rf_or_gbm["model_name"]]
        c2 = MODEL_COLORS["logreg"]

        ax1.hlines(y=y_pos, xmin=0, xmax=imp_tree[union_idx], color=c1, alpha=0.7, linewidth=1.5)
        ax1.scatter(imp_tree[union_idx], y_pos, color=c1, s=40, zorder=3)
        ax1.set_title(f"{MODEL_LABELS[rf_or_gbm['model_name']]}")
        ax1.set_xlabel("Importance (Gini)")

        ax2.hlines(y=y_pos, xmin=0, xmax=imp_lr[union_idx], color=c2, alpha=0.7, linewidth=1.5)
        ax2.scatter(imp_lr[union_idx], y_pos, color=c2, s=40, zorder=3)
        ax2.set_title("Logistic Regression")
        ax2.set_xlabel("Importance (|coef|)")

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([feat_names[i] for i in union_idx])
        ax1.invert_yaxis()

        fig.suptitle("Feature Importance Comparison — Top 20", fontsize=14, y=1.01)
        fig.tight_layout()
        _save_fig(fig, output_dir, "feature_importance_combined.png")


def plot_feature_distributions(split, output_dir):
    n_features = min(8, split.X_train.shape[1])
    feat_names = split.feature_names[:n_features]

    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8))
    axes = axes.ravel()

    X_all = np.concatenate([split.X_train, split.X_val, split.X_test])
    y_all = np.concatenate([split.y_train, split.y_val, split.y_test])

    for i, (ax, fname) in enumerate(zip(axes, feat_names)):
        for label, color, lbl_name in [(0, "tab:blue", "No Cloud"), (1, "tab:red", "Cloud")]:
            vals = X_all[y_all == label, i]
            ax.hist(vals, bins=60, alpha=0.5, color=color, label=lbl_name, density=True)
        ax.set_title(fname)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(feat_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class (first 8 raw features)", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir, "feature_distributions.png")


def plot_feature_correlation(split, output_dir):
    n_features = split.X_train.shape[1]
    if n_features > 30:
        idx = list(range(8)) + list(range(n_features - 4, n_features))
        X_sub = split.X_train[:, idx]
        names_sub = [split.feature_names[i] for i in idx]
    else:
        X_sub = split.X_train
        names_sub = split.feature_names

    corr = np.corrcoef(X_sub, rowvar=False)
    size = max(8, len(names_sub) * 0.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        corr, xticklabels=names_sub, yticklabels=names_sub,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=(len(names_sub) <= 15),
        fmt=".2f" if len(names_sub) <= 15 else "",
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Training Set)")
    fig.tight_layout()
    _save_fig(fig, output_dir, "feature_correlation.png")


def plot_class_balance(split, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    sets = {"Train": split.y_train, "Val": split.y_val, "Test": split.y_test}
    x = np.arange(len(sets))
    width = 0.35

    cloud_counts = [int(y.sum()) for y in sets.values()]
    no_cloud_counts = [int(len(y) - y.sum()) for y in sets.values()]

    ax.bar(x - width / 2, no_cloud_counts, width, label="No Cloud (0)", color="tab:blue")
    ax.bar(x + width / 2, cloud_counts, width, label="Cloud (1)", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(sets.keys())
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution Across Splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (nc, c) in enumerate(zip(no_cloud_counts, cloud_counts)):
        total = nc + c
        ax.text(i, max(nc, c) + total * 0.02, f"pos={c / total:.1%}", ha="center", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, output_dir, "class_balance.png")


def plot_calibration(results_list, split, output_dir):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    for res in results_list:
        fraction_pos, mean_predicted = calibration_curve(
            split.y_test, res["test_prob"], n_bins=10, strategy="uniform",
        )
        color = MODEL_COLORS.get(res["model_name"], None)
        ax.plot(mean_predicted, fraction_pos, "o-",
                label=MODEL_LABELS.get(res["model_name"], res["model_name"]),
                linewidth=2, markersize=5, color=color)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, output_dir, "calibration_curves.png")


def plot_spatial_predictions(results_list, split, output_dir):
    """Spatial maps of ground truth vs predicted labels on the test image.

    If split.test_all_X / test_all_coords are available, prediction panels
    cover ALL pixels (including unlabeled ones) so no white gaps appear.
    Ground truth panel still distinguishes labeled vs unlabeled pixels.
    """
    setup_plot_style()
    if split.test_coords is None:
        print("  (Skipping spatial map: no test coordinates available)")
        return

    # Determine canvas from ALL pixels if available, else from labeled only
    has_all = split.test_all_X is not None and split.test_all_coords is not None
    if has_all:
        all_coords = split.test_all_coords
        ay = all_coords[:, 0].astype(int)
        ax_ = all_coords[:, 1].astype(int)
        y_off, x_off = ay.min(), ax_.min()
        ay_rel = ay - y_off
        ax_rel = ax_ - x_off
        h, w = ay_rel.max() + 1, ax_rel.max() + 1
    else:
        y_off, x_off = 0, 0

    coords = split.test_coords
    y = coords[:, 0].astype(int)
    x = coords[:, 1].astype(int)
    if has_all:
        y_rel = y - y_off
        x_rel = x - x_off
    else:
        y_rel = y - y.min()
        x_rel = x - x.min()
        h, w = y_rel.max() + 1, x_rel.max() + 1

    n_plots = 1 + len(results_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Ground truth: labeled pixels in red/blue, unlabeled in light grey
    gt_map = np.full((h, w), np.nan)
    if has_all:
        gt_map[ay_rel, ax_rel] = 0.5  # unlabeled -> middle value (light)
    gt_map[y_rel, x_rel] = split.y_test
    axes[0].imshow(gt_map, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axes[0].set_title(f"Ground Truth\n{split.test_image_name}")
    axes[0].axis("off")

    # Prediction panels: predict ALL pixels if available
    for i, res in enumerate(results_list):
        pred_map = np.full((h, w), np.nan)
        if has_all:
            # Re-predict on all pixels using the trained model
            model = res.get("model")
            if model is not None:
                try:
                    all_prob = model.predict_proba(split.test_all_X)[:, 1]
                    all_pred = (all_prob >= res["threshold"]).astype(float)
                    pred_map[ay_rel, ax_rel] = all_pred
                except Exception:
                    # Fallback to labeled-only predictions
                    pred = (res["test_prob"] >= res["threshold"]).astype(float)
                    pred_map[y_rel, x_rel] = pred
            else:
                pred = (res["test_prob"] >= res["threshold"]).astype(float)
                pred_map[y_rel, x_rel] = pred
        else:
            pred = (res["test_prob"] >= res["threshold"]).astype(float)
            pred_map[y_rel, x_rel] = pred
        axes[i + 1].imshow(pred_map, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        f1_val = res["test_metrics"]["f1"]
        axes[i + 1].set_title(f"{res['model_name']} predictions\nF1={f1_val:.4f}")
        axes[i + 1].axis("off")

    fig.suptitle("Spatial Prediction Maps — Test Image", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir, "spatial_predictions.png")

    # Probability heatmaps - only logreg and gbm (2 panels)
    heatmap_models = [r for r in results_list if r["model_name"] in ("logreg", "gbm")]
    if not heatmap_models:
        heatmap_models = results_list[:2]  # fallback to first 2
    n_heat = len(heatmap_models)
    fig2, axes2 = plt.subplots(1, n_heat, figsize=(6 * n_heat, 5))
    if n_heat == 1:
        axes2 = [axes2]
    for i, res in enumerate(heatmap_models):
        prob_map = np.full((h, w), np.nan)
        if has_all:
            model = res.get("model")
            if model is not None:
                try:
                    all_prob = model.predict_proba(split.test_all_X)[:, 1]
                    prob_map[ay_rel, ax_rel] = all_prob
                except Exception:
                    prob_map[y_rel, x_rel] = res["test_prob"]
            else:
                prob_map[y_rel, x_rel] = res["test_prob"]
        else:
            prob_map[y_rel, x_rel] = res["test_prob"]
        im = axes2[i].imshow(prob_map, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        axes2[i].set_title(f"{MODEL_LABELS.get(res['model_name'], res['model_name'])} — P(cloud)")
        axes2[i].axis("off")
        fig2.colorbar(im, ax=axes2[i], fraction=0.046)
    fig2.suptitle("Predicted Cloud Probability Heatmaps — Test Image", fontsize=14)
    fig2.tight_layout()
    _save_fig(fig2, output_dir, "spatial_probability_heatmaps.png")


def plot_spatial_error_map(results_list, split, output_dir):
    """Per-pixel error map: correct (green), FP (red), FN (blue).
    Unlabeled pixels predicted by the model shown in light grey background.
    """
    if split.test_coords is None:
        return

    has_all = split.test_all_X is not None and split.test_all_coords is not None

    if has_all:
        all_coords = split.test_all_coords
        ay = all_coords[:, 0].astype(int)
        ax_ = all_coords[:, 1].astype(int)
        y_off, x_off = ay.min(), ax_.min()
        ay_rel = ay - y_off
        ax_rel = ax_ - x_off
        h, w = ay_rel.max() + 1, ax_rel.max() + 1
    else:
        y_off, x_off = 0, 0

    coords = split.test_coords
    y = coords[:, 0].astype(int)
    x = coords[:, 1].astype(int)
    if has_all:
        y_rel = y - y_off
        x_rel = x - x_off
    else:
        y_rel = y - y.min()
        x_rel = x - x.min()
        h, w = y_rel.max() + 1, x_rel.max() + 1

    for res in results_list:
        y_pred = (res["test_prob"] >= res["threshold"]).astype(int)
        y_true = split.y_test.astype(int)

        # 0=TN(grey), 1=TP(green), 2=FP(red), 3=FN(blue), 4=unlabeled(light grey)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#f0f0f0", "#2ca02c", "#d62728", "#1f77b4", "#e0e0e0"])

        error_map = np.full((h, w), np.nan)
        # Fill unlabeled pixels as background
        if has_all:
            error_map[ay_rel, ax_rel] = 4  # unlabeled background
        # Overlay labeled pixel errors
        error_code = np.zeros(len(y_true), dtype=int)
        error_code[(y_true == 1) & (y_pred == 1)] = 1  # TP
        error_code[(y_true == 0) & (y_pred == 1)] = 2  # FP
        error_code[(y_true == 1) & (y_pred == 0)] = 3  # FN
        error_map[y_rel, x_rel] = error_code

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(error_map, cmap=cmap, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
        ax.set_title(f"{res['model_name']} — Error Map\nGreen=TP, Red=FP, Blue=FN, Grey=TN")
        ax.axis("off")
        fig.tight_layout()
        _save_fig(fig, output_dir, f"spatial_error_map_{res['model_name']}.png")


def plot_threshold_sensitivity(results_list, split, output_dir):
    """Show how F1, precision, recall change with the decision threshold."""
    setup_plot_style()
    for res in results_list:
        prec, rec, thr = precision_recall_curve(split.y_test, res["test_prob"])
        f1_arr = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + EPS)

        model_color = MODEL_COLORS.get(res["model_name"], '#2176AE')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thr, prec[:-1], label="Precision", linewidth=1.5, color=model_color, alpha=0.6, linestyle='--')
        ax.plot(thr, rec[:-1], label="Recall", linewidth=1.5, color=model_color, alpha=0.6, linestyle=':')
        ax.plot(thr, f1_arr, label="F1", linewidth=2, color=model_color)
        ax.axvline(res["threshold"], color="gray", ls="--", alpha=0.7, label=f"chosen thr={res['threshold']:.3f}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{MODEL_LABELS.get(res['model_name'], res['model_name'])} — Threshold Sensitivity")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"threshold_sensitivity_{res['model_name']}.png")


def plot_misclassification_analysis(results_list, split, output_dir):
    """Compare feature distributions across TP / FP / FN / TN for each model."""
    key_feats = split.feature_names[:min(6, len(split.feature_names))]
    feat_idx  = {n: i for i, n in enumerate(split.feature_names)}

    for res in results_list:
        name  = res["model_name"]
        y_hat = (res["test_prob"] >= res["threshold"]).astype(int)
        y     = split.y_test

        tp = (y == 1) & (y_hat == 1)
        fp = (y == 0) & (y_hat == 1)
        fn = (y == 1) & (y_hat == 0)
        tn = (y == 0) & (y_hat == 0)

        groups = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        colors = {"TP": "#2ca02c", "FP": "#ff7f0e", "FN": "#d62728", "TN": "#1f77b4"}

        n_feats = len(key_feats)
        fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4), sharey=False)
        if n_feats == 1:
            axes = [axes]

        for ax, feat in zip(axes, key_feats):
            col = feat_idx.get(feat)
            if col is None:
                continue
            vals = split.X_test[:, col]
            for label, mask in groups.items():
                if mask.sum() > 0:
                    ax.hist(vals[mask], bins=40, alpha=0.55, density=True,
                            label=f"{label} (n={mask.sum()})",
                            color=colors[label], histtype="stepfilled")
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("Value", fontsize=8)
            ax.tick_params(labelsize=7)

        axes[0].set_ylabel("Density")
        handles = [plt.Rectangle((0,0),1,1, color=colors[g], alpha=0.55) for g in groups]
        labels  = [f"{g} (n={groups[g].sum()})" for g in groups]
        fig.legend(handles, labels, loc="upper right", fontsize=8, title="Outcome")
        fig.suptitle(f"Misclassification Analysis — {name}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        _save_fig(fig, output_dir, f"misclassification_{name}.png")


def generate_all_visualizations(results_list, split, output_dir):
    setup_plot_style()
    print(f"\n=== Generating Visualizations → {output_dir} ===")
    plot_roc_curves(results_list, split, output_dir)
    plot_pr_curves(results_list, split, output_dir)
    plot_confusion_matrices(results_list, split, output_dir)
    plot_feature_importance(results_list, split, output_dir)
    plot_feature_importance_mi(split, output_dir)
    plot_feature_distributions(split, output_dir)
    plot_feature_correlation(split, output_dir)
    plot_class_balance(split, output_dir)
    plot_calibration(results_list, split, output_dir)
    plot_spatial_predictions(results_list, split, output_dir)
    plot_spatial_error_map(results_list, split, output_dir)
    plot_threshold_sensitivity(results_list, split, output_dir)
    plot_misclassification_analysis(results_list, split, output_dir)
    plot_lasso_path(split, output_dir)
    plot_shap_analysis(results_list, split, output_dir)
    print("=== Visualization generation complete ===\n")


def plot_lasso_path(split, output_dir):
    """Lasso regularization path: show which features survive as C increases."""
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(split.X_train)
    y_tr = split.y_train

    Cs = np.logspace(-3, 2, 40)
    coef_matrix = []
    for C in Cs:
        clf = LR(penalty="l1", solver="saga", C=C, max_iter=2000, random_state=42)
        clf.fit(X_tr, y_tr)
        coef_matrix.append(clf.coef_[0])
    coef_matrix = np.array(coef_matrix)  # (n_C, n_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(coef_matrix.shape[1]):
        ax.plot(Cs, coef_matrix[:, j], linewidth=0.8)

    # label top features at the rightmost C
    top_idx = np.argsort(np.abs(coef_matrix[-1]))[-8:]
    for j in top_idx:
        ax.annotate(split.feature_names[j],
                     xy=(Cs[-1], coef_matrix[-1, j]),
                     fontsize=7, ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength C (larger = less penalty)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Lasso (L1) Regularization Path — Feature Selection")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir, "lasso_path.png")


## Ablation

def plot_ablation_bar(rows, output_dir):
    """rows: list of (mode, n_feat, f1_mean, f1_std, pr_mean, pr_std, roc_mean, roc_std)"""
    setup_plot_style()
    modes = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(modes))

    metrics = [
        ("F1", 2, 3, '#2176AE'),
        ("PR-AUC", 4, 5, '#F57C20'),
        ("ROC-AUC", 6, 7, '#3A9B5B'),
    ]

    offsets = [-0.15, 0, 0.15]
    for (label, mean_idx, std_idx, color), offset in zip(metrics, offsets):
        means = [r[mean_idx] for r in rows]
        stds = [r[std_idx] for r in rows]
        ax.errorbar(x + offset, means, yerr=stds, fmt='o', color=color,
                   markersize=7, capsize=3, capthick=1.2, linewidth=1.2, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Feature Ablation Study (Logistic Regression, 3-Split CV)")
    ax.legend()
    ax.set_ylim(0.65, 1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "ablation_feature_groups.png")


def run_ablation(data_dir, n_files, max_samples_per_image, seed, output_dir=None,
                 ae_embed_dir=None):
    print("\n=== Feature Ablation (model=logreg, 3-fold CV) ===")
    feature_modes = ["expert", "radiance", "raw_all", "enhanced_no_ae", "enhanced_no_spatial", "enhanced"]

    rows = []
    for mode in tqdm(feature_modes, desc="Ablation feature groups", leave=False):
        folds, _ = get_cross_validation_splits(
            data_dir=data_dir,
            max_samples_per_image=max_samples_per_image,
            feature_mode=mode, seed=seed,
            ae_embed_dir=ae_embed_dir,
        )

        fold_metrics = {"f1": [], "pr_auc": [], "roc_auc": []}
        for fold_info in folds:
            split = fold_info["split"]
            low, high = fit_winsor_bounds(split.X_train)
            split.X_train = apply_winsor(split.X_train, low, high)
            split.X_val = apply_winsor(split.X_val, low, high)
            split.X_test = apply_winsor(split.X_test, low, high)

            result = evaluate_model(model_name="logreg", split=split, seed=seed)
            for k in fold_metrics:
                fold_metrics[k].append(result["test_metrics"][k])

        n_feat = folds[0]["split"].X_train.shape[1]
        rows.append((
            mode, n_feat,
            np.mean(fold_metrics["f1"]), np.std(fold_metrics["f1"]),
            np.mean(fold_metrics["pr_auc"]), np.std(fold_metrics["pr_auc"]),
            np.mean(fold_metrics["roc_auc"]), np.std(fold_metrics["roc_auc"]),
        ))

    print("\nmode           n_feat   test_f1          test_pr_auc      test_roc_auc")
    for mode, n_feat, f1m, f1s, prm, prs, rocm, rocs in rows:
        print(f"{mode:<20} {n_feat:>4}   {f1m:.4f}\u00b1{f1s:.4f}   {prm:.4f}\u00b1{prs:.4f}   {rocm:.4f}\u00b1{rocs:.4f}")

    if output_dir:
        plot_ablation_bar(rows, output_dir)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive thresholding
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_threshold(y_prob, method="otsu"):
    """Estimate a decision threshold from the predicted probability distribution
    without using any labels. Works by finding the natural split point.

    Methods:
      - 'otsu': minimize intra-class variance of P(cloud) histogram
      - 'valley': find the valley between two modes in the probability histogram
    """
    if method == "otsu":
        # Discretize probabilities into 256 bins and apply Otsu's method
        hist, bin_edges = np.histogram(y_prob, bins=256, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()
        if total == 0:
            return 0.5

        best_thr, best_var = 0.5, -1
        w0, sum0 = 0, 0
        sum_total = (hist * bin_centers).sum()

        for i in range(len(hist)):
            w0 += hist[i]
            if w0 == 0:
                continue
            w1 = total - w0
            if w1 == 0:
                break
            sum0 += hist[i] * bin_centers[i]
            mu0 = sum0 / w0
            mu1 = (sum_total - sum0) / w1
            var_between = w0 * w1 * (mu0 - mu1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thr = bin_centers[i]
        return float(best_thr)

    elif method == "valley":
        # Smooth histogram and find the valley between two peaks
        from scipy.ndimage import gaussian_filter1d
        hist, bin_edges = np.histogram(y_prob, bins=100, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist.astype(float), sigma=3)

        # Find local minima in the middle region [0.2, 0.8]
        mask = (bin_centers >= 0.2) & (bin_centers <= 0.8)
        if mask.any():
            region = smoothed[mask]
            region_centers = bin_centers[mask]
            valley_idx = np.argmin(region)
            return float(region_centers[valley_idx])
        return 0.5
    else:
        raise ValueError(f"Unknown adaptive threshold method: {method}")


def evaluate_adaptive_thresholding(cv_results, all_fold_data):
    """Re-evaluate all models with adaptive thresholds and compare to fixed."""
    print("\n" + "=" * 70)
    print("=== Adaptive Thresholding Experiment ===")
    print("=" * 70)
    print(f"{'Model':<12} {'Split':<6} {'Fixed_thr':>10} {'Adapt_thr':>10} "
          f"{'Fixed_F1':>10} {'Adapt_F1':>10} {'Delta':>8}")
    print("-" * 70)

    adaptive_cv = {}
    for model_name, fold_results in cv_results.items():
        adaptive_cv[model_name] = {"fixed_f1": [], "adaptive_f1": [],
                                    "fixed_thr": [], "adaptive_thr": []}
        for fold_idx, result in enumerate(fold_results):
            test_prob = result["test_prob"]
            y_test = all_fold_data[fold_idx][2].y_test  # split.y_test

            fixed_thr = result["threshold"]
            adapt_thr = adaptive_threshold(test_prob, method="otsu")

            fixed_metrics = compute_metrics(y_test, test_prob, fixed_thr)
            adapt_metrics = compute_metrics(y_test, test_prob, adapt_thr)

            delta = adapt_metrics["f1"] - fixed_metrics["f1"]
            print(f"{model_name:<12} {fold_idx+1:<6} {fixed_thr:>10.4f} {adapt_thr:>10.4f} "
                  f"{fixed_metrics['f1']:>10.4f} {adapt_metrics['f1']:>10.4f} {delta:>+8.4f}")

            adaptive_cv[model_name]["fixed_f1"].append(fixed_metrics["f1"])
            adaptive_cv[model_name]["adaptive_f1"].append(adapt_metrics["f1"])
            adaptive_cv[model_name]["fixed_thr"].append(fixed_thr)
            adaptive_cv[model_name]["adaptive_thr"].append(adapt_thr)

    print("\n--- Summary (mean across splits) ---")
    print(f"{'Model':<12} {'Fixed_F1':>10} {'Adapt_F1':>10} {'Delta':>8}")
    for model_name in adaptive_cv:
        fm = np.mean(adaptive_cv[model_name]["fixed_f1"])
        am = np.mean(adaptive_cv[model_name]["adaptive_f1"])
        print(f"{model_name:<12} {fm:>10.4f} {am:>10.4f} {am-fm:>+8.4f}")

    return adaptive_cv


# ═══════════════════════════════════════════════════════════════════════════════
# Feature selection experiment (top-K SHAP features)
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_selection_experiment(all_fold_data, cv_results, feature_names,
                                     top_k_list=None, seed=42):
    """Train models using only the top-K most important features (by SHAP)
    and compare to the full feature set.
    """
    if top_k_list is None:
        top_k_list = [10, 15, 20, 30]

    print("\n" + "=" * 70)
    print("=== Feature Selection Experiment (SHAP-based) ===")
    print("=" * 70)

    # Get SHAP importance from GBM on fold 1
    # Use the first fold's GBM result for feature ranking
    gbm_results = cv_results.get("gbm", [])
    if not gbm_results:
        print("  No GBM results available for SHAP. Skipping.")
        return None

    gbm_model = gbm_results[0]["model"]
    fold1_split = all_fold_data[0][2]  # split

    try:
        import shap
        explainer = shap.TreeExplainer(gbm_model)
        # Use a subsample for speed
        n_sample = min(5000, len(fold1_split.X_test))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(fold1_split.X_test), size=n_sample, replace=False)
        shap_values = explainer.shap_values(fold1_split.X_test[idx])
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_ranking = np.argsort(mean_abs_shap)[::-1]
    except Exception as e:
        print(f"  SHAP failed ({e}), falling back to GBM feature_importances_")
        importances = gbm_model.feature_importances_
        feature_ranking = np.argsort(importances)[::-1]
        mean_abs_shap = importances

    print("\nTop-20 features by importance:")
    for i in range(min(20, len(feature_ranking))):
        fi = feature_ranking[i]
        fname = feature_names[fi] if fi < len(feature_names) else f"f{fi}"
        print(f"  {i+1:>3}. {fname:<25} importance={mean_abs_shap[fi]:.4f}")

    # Evaluate with different top-K
    results_table = []
    for top_k in top_k_list:
        selected_idx = feature_ranking[:top_k]

        fold_f1s = []
        fold_roc = []
        for fold_idx, (fold_info, fold_results, split, low, high) in enumerate(all_fold_data):
            # Subset features
            X_train_sel = split.X_train[:, selected_idx]
            X_val_sel = split.X_val[:, selected_idx]
            X_test_sel = split.X_test[:, selected_idx]

            sub_split = DatasetSplit(
                X_train=X_train_sel, y_train=split.y_train,
                X_val=X_val_sel, y_val=split.y_val,
                X_test=X_test_sel, y_test=split.y_test,
                feature_names=[feature_names[i] for i in selected_idx if i < len(feature_names)],
            )

            result = evaluate_model(model_name="gbm", split=sub_split, seed=seed)
            fold_f1s.append(result["test_metrics"]["f1"])
            fold_roc.append(result["test_metrics"]["roc_auc"])

        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        mean_roc = np.mean(fold_roc)
        results_table.append((top_k, mean_f1, std_f1, mean_roc))

    # Also get full feature set results for comparison
    full_f1 = np.mean([r["test_metrics"]["f1"] for r in cv_results["gbm"]])
    full_roc = np.mean([r["test_metrics"]["roc_auc"] for r in cv_results["gbm"]])
    n_full = all_fold_data[0][2].X_train.shape[1]

    print(f"\n{'Features':>10} {'Mean F1':>12} {'Std F1':>10} {'Mean ROC':>12}")
    print("-" * 48)
    for top_k, mf1, sf1, mroc in results_table:
        marker = " <--" if mf1 >= full_f1 else ""
        print(f"{top_k:>10} {mf1:>12.4f} {sf1:>10.4f} {mroc:>12.4f}{marker}")
    print(f"{'ALL ('+str(n_full)+')':>10} {full_f1:>12.4f} {'':>10} {full_roc:>12.4f}  (baseline)")

    return results_table


# ═══════════════════════════════════════════════════════════════════════════════
# Predict on ALL images (labeled + unlabeled)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_all_images(model, threshold, feature_mode, data_dir, predictions_dir,
                       winsor_low=None, winsor_high=None, ae_embed_dir=None):
    """
    Run the trained classifier on every .npz file (labeled and unlabeled).
    Saves per-image CSVs and a summary CSV with cloud-coverage statistics.
    """
    resolved_data_dir, _ = resolve_data_dir(data_dir)
    all_files = sorted(glob.glob(os.path.join(resolved_data_dir, "*.npz")))
    os.makedirs(predictions_dir, exist_ok=True)

    print(f"\n=== Predicting on {len(all_files)} images → {predictions_dir} ===")
    summary_rows = []

    for fp in tqdm(all_files, desc="Predicting images"):
        arr = load_npz_array(fp)
        has_label = (arr.shape[1] == 11)
        coords = arr[:, :2]

        basename = os.path.splitext(os.path.basename(fp))[0]
        sp_lookup, sp_names = compute_spatial_features(arr)
        X, _ = build_features(
            arr, mode=feature_mode,
            ae_embed_dir=ae_embed_dir, image_basename=os.path.basename(fp),
            spatial_lookup=sp_lookup, spatial_names=sp_names,
        )

        if winsor_low is not None and winsor_high is not None:
            if X.shape[1] == len(winsor_low):
                X = apply_winsor(X, winsor_low, winsor_high)

        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)

        df = pd.DataFrame({
            "y": coords[:, 0].astype(int),
            "x": coords[:, 1].astype(int),
            "cloud_prob": np.round(prob, 6),
            "cloud_pred": pred,
        })

        if has_label:
            raw_labels = arr[:, 10].astype(int)
            df["expert_label"] = raw_labels

        df.to_csv(os.path.join(predictions_dir, f"{basename}_pred.csv"), index=False)

        n_pixels = len(pred)
        n_cloud = int(pred.sum())
        cloud_frac = n_cloud / n_pixels if n_pixels > 0 else 0.0
        summary_rows.append({
            "image": basename,
            "n_pixels": n_pixels,
            "n_cloud": n_cloud,
            "n_clear": n_pixels - n_cloud,
            "cloud_fraction": round(cloud_frac, 4),
            "has_expert_label": has_label,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(predictions_dir, "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")
    print(f"  Cloud fraction range: {summary_df['cloud_fraction'].min():.3f} – "
          f"{summary_df['cloud_fraction'].max():.3f}")
    print(f"  Mean cloud fraction:  {summary_df['cloud_fraction'].mean():.3f}")
    return summary_df


def plot_prediction_gallery(model_name, threshold, feature_mode, data_dir,
                            output_dir, n_samples=8,
                            winsor_low=None, winsor_high=None, seed=42,
                            ae_embed_dir=None):
    """
    Visualise spatial cloud predictions for a sample of unlabeled images.
    """
    resolved_data_dir, _ = resolve_data_dir(data_dir)
    all_files = sorted(glob.glob(os.path.join(resolved_data_dir, "*.npz")))

    labeled_files = set()
    unlabeled_files = []
    for fp in all_files:
        arr = load_npz_array(fp)
        if arr.shape[1] == 11:
            labeled_files.add(fp)
        else:
            unlabeled_files.append(fp)

    rng = np.random.default_rng(seed)
    n_show = min(n_samples, len(unlabeled_files))
    if n_show == 0:
        print("  (No unlabeled images to show in gallery)")
        return
    chosen = rng.choice(len(unlabeled_files), size=n_show, replace=False)
    chosen_files = [unlabeled_files[i] for i in sorted(chosen)]

    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig_pred, axes_pred = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    fig_prob, axes_prob = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_pred = np.array(axes_pred).ravel() if n_show > 1 else [axes_pred]
    axes_prob = np.array(axes_prob).ravel() if n_show > 1 else [axes_prob]


    for idx, fp in enumerate(chosen_files):
        arr = load_npz_array(fp)
        coords = arr[:, :2]
        sp_lookup, sp_names = compute_spatial_features(arr)
        X, _ = build_features(
            arr, mode=feature_mode,
            ae_embed_dir=ae_embed_dir, image_basename=os.path.basename(fp),
            spatial_lookup=sp_lookup, spatial_names=sp_names,
        )
        if winsor_low is not None and winsor_high is not None:
            X = apply_winsor(X, winsor_low, winsor_high)

        prob = _gallery_model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)

        y_c = coords[:, 0].astype(int)
        x_c = coords[:, 1].astype(int)
        yr = y_c - y_c.min()
        xr = x_c - x_c.min()
        h, w = yr.max() + 1, xr.max() + 1

        pred_map = np.full((h, w), np.nan)
        pred_map[yr, xr] = pred
        prob_map = np.full((h, w), np.nan)
        prob_map[yr, xr] = prob

        basename = os.path.splitext(os.path.basename(fp))[0]
        cloud_frac = pred.mean()

        axes_pred[idx].imshow(pred_map, cmap="RdBu_r", vmin=0, vmax=1,
                              aspect="auto", interpolation="nearest")
        axes_pred[idx].set_title(f"{basename}\ncloud={cloud_frac:.1%}", fontsize=9)
        axes_pred[idx].axis("off")

        axes_prob[idx].imshow(prob_map, cmap="hot_r", vmin=0, vmax=1,
                                   aspect="auto", interpolation="nearest")
        axes_prob[idx].set_title(f"{basename}\nP(cloud)", fontsize=9)
        axes_prob[idx].axis("off")

    for j in range(n_show, len(axes_pred)):
        axes_pred[j].set_visible(False)
        axes_prob[j].set_visible(False)

    fig_pred.suptitle(f"Unlabeled Image Predictions ({model_name}, thr={threshold:.3f})", fontsize=13)
    fig_pred.tight_layout()
    _save_fig(fig_pred, output_dir, f"unlabeled_predictions_{model_name}.png")

    fig_prob.suptitle(f"Unlabeled Image Cloud Probabilities ({model_name})", fontsize=13)
    fig_prob.tight_layout()
    _save_fig(fig_prob, output_dir, f"unlabeled_probabilities_{model_name}.png")


_gallery_model = None  # set by run_advanced_mode before calling plot_prediction_gallery


def plot_cloud_fraction_histogram(summary_df, output_dir):
    """Histogram of cloud fraction across all 164 images."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labeled_mask = summary_df["has_expert_label"]

    ax.hist(summary_df.loc[~labeled_mask, "cloud_fraction"], bins=30, alpha=0.7,
            color="steelblue", label=f"Unlabeled ({(~labeled_mask).sum()} images)", edgecolor="white")
    ax.hist(summary_df.loc[labeled_mask, "cloud_fraction"], bins=10, alpha=0.8,
            color="tab:red", label=f"Labeled ({labeled_mask.sum()} images)", edgecolor="white")
    ax.set_xlabel("Predicted Cloud Fraction")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Predicted Cloud Coverage Across All Images")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir, "cloud_fraction_histogram.png")


## Cross-validation summary

def print_cv_summary(cv_results):
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION SUMMARY (Test Set Metrics)")
    print(f"{'=' * 80}")

    metrics_keys = ["f1", "roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall"]
    for metric in metrics_keys:
        print(f"\n  {metric.upper()}")
        header = f"    {'Model':<12}"
        for i in range(3):
            header += f" {'Split ' + str(i + 1):>8}"
        header += f" {'Mean':>8} {'Std':>8}"
        print(header)
        print(f"    {'-' * 56}")
        for model_name, fold_results in cv_results.items():
            vals = [r["test_metrics"][metric] for r in fold_results]
            row = f"    {model_name:<12}"
            for v in vals:
                row += f" {v:>8.4f}"
            row += f" {np.mean(vals):>8.4f} {np.std(vals):>8.4f}"
            print(row)


def plot_cv_summary(cv_results, output_dir):
    """Cleveland dot plot of mean CV metrics with std-dev error bars."""
    setup_plot_style()
    metrics = ["f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    metric_labels = ["F1", "ROC-AUC", "PR-AUC", "Balanced Accuracy"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=True)

    model_names = list(cv_results.keys())
    y_pos = np.arange(len(model_names))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        for i, mname in enumerate(model_names):
            vals = [r["test_metrics"][metric] for r in cv_results[mname]]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            color = MODEL_COLORS.get(mname, '#333')

            ax.errorbar(mean_val, i, xerr=std_val, fmt='o', color=color,
                       markersize=8, capsize=4, capthick=1.5, linewidth=1.5,
                       label=MODEL_LABELS.get(mname, mname) if ax == axes[0] else None)

        ax.set_title(label, fontsize=12)
        ax.set_yticks(y_pos)
        if ax == axes[0]:
            ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in model_names])
        ax.set_xlim(0.6, 1.02)
        ax.axvline(x=0.5, color='grey', linestyle=':', alpha=0.3)

    fig.suptitle("3-Split Cross-Validation Results (Mean \u00b1 Std)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "cv_summary.png")


def plot_cv_fold_comparison(cv_results, output_dir):
    """Line plot showing each model's test F1 across the 3 folds."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    fold_x = [1, 2, 3]
    for model_name, fold_results in cv_results.items():
        f1_vals = [r["test_metrics"]["f1"] for r in fold_results]
        color = MODEL_COLORS.get(model_name, None)
        ax.plot(fold_x, f1_vals, "o-",
                label=MODEL_LABELS.get(model_name, model_name),
                linewidth=2, markersize=8, color=color)
    ax.set_xlabel("Split")
    ax.set_ylabel("Test F1 Score")
    ax.set_title("Model Performance Across CV Splits")
    ax.set_xticks(fold_x)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir, "cv_fold_comparison.png")


## EDA: per-image distribution shift

def plot_distribution_shift(data_dir, output_dir):
    """KDE/histogram of key features per labeled image — reveals distribution shift across folds."""
    resolved_data_dir, _ = resolve_data_dir(data_dir)
    labeled_files = discover_labeled_files(resolved_data_dir)

    key_feats = ["NDAI", "SD", "CORR", "Rad_AN"]
    feat_col  = {n: i + 2 for i, n in enumerate(RAW_FEATURE_NAMES)}  # +2 for y,x
    colors    = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, len(key_feats), figsize=(5 * len(key_feats), 8))
    names_used = []

    for ci, fp in enumerate(labeled_files):
        arr  = load_npz_array(fp)
        mask = arr[:, 10].astype(int) != 0
        lbl  = (arr[mask, 10].astype(int) == 1)
        img_name = os.path.splitext(os.path.basename(fp))[0]
        names_used.append(img_name)

        for ax_row, cloud_flag, title_sfx in [(axes[0], slice(None), "all"), (axes[1], lbl, "cloud")]:
            for ax, feat in zip(ax_row, key_feats):
                vals = arr[mask, feat_col[feat]][cloud_flag] if title_sfx == "cloud" else arr[mask, feat_col[feat]]
                ax.hist(vals, bins=60, alpha=0.5, density=True,
                        color=colors[ci], label=img_name, histtype="stepfilled")

    for row_axes, title_sfx in [(axes[0], "all labeled"), (axes[1], "cloud only")]:
        for ax, feat in zip(row_axes, key_feats):
            ax.set_title(f"{feat} ({title_sfx})", fontsize=10)
            ax.set_xlabel("Value")
            ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel("Density")
    axes[1][0].set_ylabel("Density")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(names_used))]
    fig.legend(handles, names_used, loc="upper right", fontsize=9, title="Image")
    fig.suptitle("Feature Distribution Shift Across Labeled Images", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    _save_fig(fig, output_dir, "distribution_shift.png")


## EDA: mutual information + ROC-AUC feature importance

def plot_shap_analysis(results_list, split, output_dir):
    """SHAP global bar + beeswarm for tree models; waterfall for top misclassified pixels."""
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — skipping")
        return

    tree_models = [r for r in results_list if r["model_name"] in ("rf", "gbm")]
    if not tree_models:
        print("  [SHAP] no tree models found — skipping")
        return

    # Use a sample for speed (SHAP on full test set can be slow)
    max_bg   = min(500, len(split.X_train))
    max_test = min(300, len(split.X_test))
    rng = np.random.default_rng(42)
    bg_idx   = rng.choice(len(split.X_train), max_bg,  replace=False)
    te_idx   = rng.choice(len(split.X_test),  max_test, replace=False)
    X_bg  = split.X_train[bg_idx]
    X_te  = split.X_test[te_idx]
    y_te  = split.y_test[te_idx]

    for res in tree_models:
        name  = res["model_name"]
        model = res["model"]
        print(f"  [SHAP] computing for {name} ...")

        try:
            explainer   = shap.TreeExplainer(model, data=X_bg, feature_names=split.feature_names)
            shap_values = explainer(X_te)          # Explanation object
            sv = shap_values.values                # (n, p) — class-1 values for tree classifiers
            if sv.ndim == 3:                       # multi-output: take class 1
                sv = sv[:, :, 1]
        except Exception as e:
            print(f"    TreeExplainer failed ({e}), falling back to KernelExplainer")
            explainer   = shap.KernelExplainer(model.predict_proba, X_bg)
            sv          = explainer.shap_values(X_te, nsamples=100)[1]

        feat_names = split.feature_names

        ## Global bar chart
        mean_abs = np.abs(sv).mean(axis=0)
        top_n    = min(20, len(feat_names))
        order    = np.argsort(mean_abs)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        ax.barh(range(top_n), mean_abs[order], color="#1f77b4", alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Global Feature Importance — {name}", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"shap_bar_{name}.png")

        ## Beeswarm
        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
        feat_order = [feat_names[i] for i in order]
        sv_ordered = sv[:, order]
        x_ordered  = X_te[:, order]
        # Normalise feature values to [0,1] for colour mapping
        x_norm = (x_ordered - x_ordered.min(axis=0)) / (x_ordered.max(axis=0) - x_ordered.min(axis=0) + 1e-9)
        cmap = plt.cm.coolwarm

        for row_i in range(top_n):
            y_vals = sv_ordered[:, row_i]
            colors = cmap(x_norm[:, row_i])
            jitter = np.random.default_rng(row_i).uniform(-0.3, 0.3, len(y_vals))
            ax.scatter(y_vals, np.full(len(y_vals), row_i) + jitter,
                       c=colors, alpha=0.4, s=6, linewidths=0)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_order, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_title(f"SHAP Beeswarm — {name}\n(blue=low feature value, red=high)", fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Feature value (normalised)", shrink=0.6)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"shap_beeswarm_{name}.png")

        ## Waterfall for most-confused FP and FN pixels
        y_hat = (res["test_prob"][te_idx] >= res["threshold"]).astype(int)
        fp_idx = np.where((y_te == 0) & (y_hat == 1))[0]
        fn_idx = np.where((y_te == 1) & (y_hat == 0))[0]

        for case_label, idx_pool in [("FP", fp_idx), ("FN", fn_idx)]:
            if len(idx_pool) == 0:
                continue
            # Pick the pixel with the highest |SHAP sum| in that group (most extreme)
            pick = idx_pool[np.abs(sv[idx_pool]).sum(axis=1).argmax()]
            sv_px   = sv[pick][order[:top_n]]
            base    = sv[pick].sum() - sv_px.sum()          # crude base
            cumsum  = np.cumsum(sv_px)
            starts  = np.concatenate([[base], base + cumsum[:-1]])

            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
            colors  = ["#d62728" if v > 0 else "#1f77b4" for v in sv_px]
            ax.barh(range(top_n), sv_px, left=starts, color=colors, alpha=0.85, height=0.6)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(feat_order, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value")
            ax.set_title(f"SHAP Waterfall — {name} | {case_label} pixel\n"
                         f"(red=pushes toward cloud, blue=pushes away)", fontsize=10)
            ax.invert_yaxis()
            fig.tight_layout()
            _save_fig(fig, output_dir, f"shap_waterfall_{name}_{case_label}.png")


def plot_feature_importance_mi(split, output_dir):
    """Univariate MI and ROC-AUC importance — complements model-based importance."""
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(split.X_train, split.y_train, random_state=42)

    roc_aucs = []
    for j in range(split.X_train.shape[1]):
        try:
            score = roc_auc_score(split.y_train, split.X_train[:, j])
            roc_aucs.append(max(score, 1.0 - score))
        except Exception:
            roc_aucs.append(0.5)
    roc_aucs = np.array(roc_aucs)

    top_n  = min(20, len(split.feature_names))
    order  = np.argsort(mi)[::-1][:top_n]
    names  = [split.feature_names[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_n * 0.35)))
    for ax, vals, xlabel, color, fname in [
        (axes[0], mi[order],       "Mutual Information",       "#1f77b4", "feature_importance_mi_roc.png"),
        (axes[1], roc_aucs[order], "ROC-AUC (univariate)",     "#ff7f0e", None),
    ]:
        ax.barh(range(top_n), vals, color=color)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_xlim(0.5, 1.0)
    fig.suptitle("Univariate Feature Importance (MI & ROC-AUC)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir, "feature_importance_mi_roc.png")


## EDA: global Moran's I (spatial autocorrelation)

def _morans_i_raster(grid, valid_mask):
    """Efficient rook-contiguity Moran's I for a dense raster grid."""
    z = np.where(valid_mask, grid - grid[valid_mask].mean(), 0.0)
    cross = (np.sum(z[1:, :] * z[:-1, :]) + np.sum(z[:, 1:] * z[:, :-1])) * 2
    v = valid_mask.astype(np.float64)
    W = (np.sum(v[1:, :] * v[:-1, :]) + np.sum(v[:, 1:] * v[:, :-1])) * 2
    denom = np.sum(z ** 2)
    if denom == 0 or W == 0:
        return np.nan
    return float(valid_mask.sum() / W * cross / denom)


def plot_morans_i(data_dir, output_dir):
    """Bar chart of global Moran's I per feature per labeled image.

    High Moran's I (> 0) confirms strong spatial autocorrelation, justifying
    image-level (rather than pixel-level) train/test splits.
    """
    resolved_data_dir, _ = resolve_data_dir(data_dir)
    labeled_files = discover_labeled_files(resolved_data_dir)

    key_feats = ["NDAI", "SD", "CORR", "Rad_AN", "label"]
    feat_col  = {n: i + 2 for i, n in enumerate(RAW_FEATURE_NAMES)}
    feat_col["label"] = 10

    results = {}
    for fp in labeled_files:
        arr      = load_npz_array(fp)
        img_name = os.path.splitext(os.path.basename(fp))[0]
        y = arr[:, 0].astype(int)
        x = arr[:, 1].astype(int)
        miny, minx = y.min(), x.min()
        H = int(y.max() - miny + 1)
        W = int(x.max() - minx + 1)
        valid = np.zeros((H, W), dtype=bool)
        valid[y - miny, x - minx] = True

        mis = []
        for feat in key_feats:
            col = feat_col[feat]
            if col >= arr.shape[1]:
                mis.append(np.nan)
                continue
            vals = arr[:, col].astype(float)
            if feat == "label":
                mask_lbl = vals != 0
                vals = np.where(mask_lbl, (vals == 1).astype(float), np.nan)
            grid = np.full((H, W), np.nan)
            grid[y - miny, x - minx] = vals
            v_mask = valid & ~np.isnan(grid)
            grid = np.where(v_mask, grid, 0.0)
            mis.append(_morans_i_raster(grid, v_mask))
        results[img_name] = mis

    img_names = list(results.keys())
    x_pos = np.arange(len(key_feats))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, img_name in enumerate(img_names):
        ax.bar(x_pos + i * width, results[img_name], width,
               label=img_name, color=colors[i], alpha=0.85)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(key_feats)
    ax.set_ylabel("Global Moran's I")
    ax.set_title("Spatial Autocorrelation (Moran's I) per Feature and Image\n"
                 "Values > 0 indicate spatial clustering → justifies image-level CV split")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_dir, "morans_i.png")


## Quick baseline mode

def run_quick_mode(seed):
    print("=== Quick Modeling Start ===")
    X, y, meta = make_labeled_table_low_ram(winsorize=True, winsor_quantiles=(0.01, 0.99))
    print(f"samples: {meta['n_samples']:,}, features: {meta['n_features']}, pos_rate: {meta['positive_rate']:.3f}")
    print(f"features: {meta['feature_names']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n=== Baseline Results (Logistic Regression) ===")
    print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"roc_auc:  {roc_auc_score(y_test, y_prob):.4f}")
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, digits=4))


## Advanced mode

def run_advanced_mode(args):
    global _gallery_model

    cap_label = args.max_samples_per_image or "ALL"
    ae_dir = getattr(args, "ae_embed_dir", None) or None
    print("=== Full Modeling Pipeline with 3-Fold Image-Level Cross-Validation ===")
    print(f"max_samples_per_image={cap_label}, feature_mode={args.feature_mode}")
    print(f"Models: {args.models}")
    print(f"AE embeddings: {ae_dir or 'disabled'}")

    ## Phase 1: 3-fold CV across labeled images
    folds, feature_names = get_cross_validation_splits(
        data_dir=args.data_dir,
        max_samples_per_image=args.max_samples_per_image,
        feature_mode=args.feature_mode,
        seed=args.seed,
        ae_embed_dir=ae_dir,
    )

    cv_results = {m: [] for m in args.models}
    all_fold_data = []

    for fold_info in folds:
        fold_num = fold_info["fold_idx"] + 1
        split = fold_info["split"]

        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_num}: Train={fold_info['train_image']}, "
              f"Val={fold_info['val_image']}, Test={fold_info['test_image']}")
        print(f"{'=' * 70}")
        print(f"  Train: {split.X_train.shape}, pos_rate={split.y_train.mean():.3f}")
        print(f"  Val:   {split.X_val.shape}, pos_rate={split.y_val.mean():.3f}")
        print(f"  Test:  {split.X_test.shape}, pos_rate={split.y_test.mean():.3f}")

        low, high = fit_winsor_bounds(split.X_train)
        split.X_train = apply_winsor(split.X_train, low, high)
        split.X_val = apply_winsor(split.X_val, low, high)
        split.X_test = apply_winsor(split.X_test, low, high)
        if split.test_all_X is not None:
            split.test_all_X = apply_winsor(split.test_all_X, low, high)

        fold_results = []
        for model_name in tqdm(args.models, desc=f"Fold {fold_num} models", leave=False):
            result = evaluate_model(model_name=model_name, split=split, seed=args.seed)
            fold_results.append(result)
            cv_results[model_name].append(result)
            print(f"\n  --- {model_name} ---")
            print_metrics_table("    Validation", result["val_metrics"])
            print_metrics_table("    Test", result["test_metrics"])

        all_fold_data.append((fold_info, fold_results, split, low, high))

        if args.output_dir:
            fold_dir = os.path.join(args.output_dir, f"fold_{fold_num}")
            generate_all_visualizations(fold_results, split, fold_dir)

    ## Phase 2: CV summary
    print_cv_summary(cv_results)

    output_dir = args.output_dir
    if output_dir:
        plot_cv_summary(cv_results, output_dir)
        plot_cv_fold_comparison(cv_results, output_dir)
        plot_distribution_shift(args.data_dir, output_dir)
        plot_morans_i(args.data_dir, output_dir)

    if args.run_ablation:
        run_ablation(
            data_dir=args.data_dir, n_files=args.n_files,
            max_samples_per_image=args.max_samples_per_image,
            seed=args.seed, output_dir=output_dir,
            ae_embed_dir=args.ae_embed_dir,
        )

    ## Phase 2b: Adaptive thresholding experiment
    evaluate_adaptive_thresholding(cv_results, all_fold_data)

    ## Phase 2c: Feature selection experiment
    run_feature_selection_experiment(
        all_fold_data, cv_results, feature_names,
        top_k_list=[10, 15, 20, 30, 50],
        seed=args.seed,
    )

    ## Phase 3: Predict on ALL images using best model
    best_model_name = max(
        cv_results.keys(),
        key=lambda m: np.mean([r["test_metrics"]["f1"] for r in cv_results[m]]),
    )
    best_mean_f1 = np.mean([r["test_metrics"]["f1"] for r in cv_results[best_model_name]])
    best_fold_result = max(
        cv_results[best_model_name], key=lambda r: r["test_metrics"]["f1"],
    )
    best_fold_idx = cv_results[best_model_name].index(best_fold_result)
    _, _, _, best_low, best_high = all_fold_data[best_fold_idx]

    print(f"\n>>> Best model: {best_model_name} "
          f"(mean test F1={best_mean_f1:.4f}, "
          f"best fold test F1={best_fold_result['test_metrics']['f1']:.4f})")

    predictions_dir = args.predictions_dir
    if predictions_dir:
        summary_df = predict_all_images(
            model=best_fold_result["model"],
            threshold=best_fold_result["threshold"],
            feature_mode=args.feature_mode,
            data_dir=args.data_dir,
            predictions_dir=predictions_dir,
            winsor_low=best_low, winsor_high=best_high,
            ae_embed_dir=ae_dir,
        )

        if output_dir:
            plot_cloud_fraction_histogram(summary_df, output_dir)

            _gallery_model = best_fold_result["model"]
            plot_prediction_gallery(
                model_name=best_model_name,
                threshold=best_fold_result["threshold"],
                feature_mode=args.feature_mode,
                data_dir=args.data_dir,
                output_dir=output_dir, n_samples=8,
                winsor_low=best_low, winsor_high=best_high,
                seed=args.seed,
                ae_embed_dir=ae_dir,
            )
            _gallery_model = None


## CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cloud detection modeling with full visualizations.",
    )
    parser.add_argument(
        "--mode", type=str, default="advanced", choices=["quick", "advanced"],
        help="quick: baseline logreg; advanced: image-level split + ablation + visualizations.",
    )
    parser.add_argument("--data-dir", type=str, default="../data", help="Path to .npz folder")
    parser.add_argument("--n-files", type=int, default=3, help="How many labeled files to use")
    parser.add_argument(
        "--max-samples-per-image", type=int, default=0,
        help="Cap samples per image (0 = no cap, use all labelled pixels)",
    )
    parser.add_argument(
        "--feature-mode", type=str, default="enhanced",
        choices=["expert", "radiance", "raw_all", "enhanced", "enhanced_no_ae", "enhanced_no_spatial"],
        help="Feature set to train with",
    )
    parser.add_argument(
        "--models", nargs="+", default=["logreg", "rf", "gbm", "knn"],
        choices=["logreg", "rf", "gbm", "knn"],
        help="Models to evaluate (default: all four)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ablation", action="store_true", help="Run feature-group ablation")
    parser.add_argument(
        "--output-dir", type=str, default="../report/figures",
        help="Directory to save visualization plots",
    )
    parser.add_argument(
        "--predictions-dir", type=str, default="../data/predictions",
        help="Directory to save per-image prediction CSVs (set to '' to skip)",
    )
    parser.add_argument(
        "--log-file", type=str, default="",
        help="Optional path to save all console output.",
    )
    parser.add_argument(
        "--ae-embed-dir", type=str, default="",
        help="Directory containing *_ae.csv autoencoder embeddings. "
             "If provided (and files exist), AE features are appended to the feature matrix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_samples_per_image <= 0:
        args.max_samples_per_image = None

    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_handle = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_handle)
        sys.stderr = Tee(sys.stderr, log_handle)
        print(f"Logging output to: {args.log_file}")

    try:
        if args.mode == "quick":
            run_quick_mode(seed=args.seed)
        else:
            run_advanced_mode(args)
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
