"""
ensemble.py — Train and analyze RF + HGB ensemble models.

Outputs:
  models   → results/
  HGB      → results/hgb/
  RF       → results/rf/
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde, skew
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from logreg_svm import plot_labeled_vs_unlabeled, plot_spatial_errors

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "feature_eng_dataset"
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
HGB_OUT     = RESULTS_DIR / "hgb"
RF_OUT      = RESULTS_DIR / "rf"
for d in (RESULTS_DIR, HGB_OUT, RF_OUT):
    d.mkdir(parents=True, exist_ok=True)

# ── Feature sets ───────────────────────────────────────────────────────────────

BASE_FEATURES     = ["SD", "CORR", "NDAI_DF_AF"]
ORIGINAL_FEATURES = ["SD", "CORR", "DF", "CF", "BF", "AF", "AN", "NDAI_DF_AF"]
AE_FEATURES       = [f"ae{i}" for i in range(32)]
FULL_FEATURES     = ORIGINAL_FEATURES + AE_FEATURES
TRAIN_FEATURES    = FULL_FEATURES  # keep train/eval features consistent

LABELED_IDS = {"O012791", "O013257", "O013490"}

# ── Model specs (for HGB comparative analysis) ────────────────────────────────

def model_specs():
    return {
        "rf_base": {
            "model": RandomForestClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            "features": BASE_FEATURES,
            "family": "tree",
        },
        "rf_full": {
            "model": RandomForestClassifier(
                n_estimators=350, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            "features": FULL_FEATURES,
            "family": "tree",
        },
        "hgb_original": {
            "model": HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=350, max_depth=8,
                min_samples_leaf=40, l2_regularization=1e-3,
                validation_fraction=0.15, n_iter_no_change=20, random_state=42,
            ),
            "features": ORIGINAL_FEATURES,
            "family": "boosting",
        },
        "hgb_full": {
            "model": HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=350, max_depth=8,
                min_samples_leaf=40, l2_regularization=1e-3,
                validation_fraction=0.15, n_iter_no_change=20, random_state=42,
            ),
            "features": FULL_FEATURES,
            "family": "boosting",
        },
    }

# RF label-flip params
RF_FLIP_PARAMS = dict(min_samples_leaf=10, max_features="sqrt", bootstrap=True, n_jobs=-1)
RF_FLIP_RATE   = 0.05
RF_FLIP_REPS   = 3

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    df_train = pd.read_csv(DATASET_DIR / "train_features_opt.csv")
    df_test  = pd.read_csv(DATASET_DIR / "test_features_opt.csv")
    print(f"Train: {df_train.shape}  Test: {df_test.shape}")
    print(f"Train label counts:\n{df_train['label'].value_counts(dropna=False).sort_index()}")
    print(f"Test label counts:\n{df_test['label'].value_counts(dropna=False).sort_index()}")
    return df_train, df_test

# ── Shared metric helper ───────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall":    recall_score(y_true, y_pred, pos_label=1),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Training
# ══════════════════════════════════════════════════════════════════════════════

def train_rf(X_train, y_train):
    print("Training Random Forest (fixed parameters)...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=20,
        min_samples_leaf=10, max_features="sqrt", ccp_alpha=0.0001,
        bootstrap=True, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_hgb(X_train, y_train):
    print("Training HistGradientBoosting (fixed parameters)...")
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=350, max_depth=8,
        min_samples_leaf=40, l2_regularization=1e-3,
        validation_fraction=0.15, n_iter_no_change=20, random_state=42,
    )
    hgb.fit(X_train, y_train)
    return hgb


def evaluate_and_save(model, df_test, model_name):
    X_test, y_test = df_test[TRAIN_FEATURES], df_test["label"]
    y_pred = model.predict(X_test)

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Expected positive class label 1 in model classes, got {classes}")
    pos_idx = classes.index(1)
    y_prob = model.predict_proba(X_test)[:, pos_idx]

    m = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "ROC AUC":   roc_auc_score(y_test, y_prob),
        "F1 Score":  f1_score(y_test, y_pred, pos_label=1),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall":    recall_score(y_test, y_pred, pos_label=1),
    }
    print(f"\n--- {model_name} ---")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, RESULTS_DIR / f"best_{model_name}_model.pkl")
    pd.DataFrame([m]).to_csv(RESULTS_DIR / f"{model_name}_metrics.csv", index=False)

    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(model.feature_importances_, index=TRAIN_FEATURES).sort_values(ascending=False)
        feat_imp.to_csv(RESULTS_DIR / f"{model_name}_feature_importance.csv")
        print(f"\nTop 10 {model_name} Feature Importances:\n{feat_imp.head(10)}")

    return m, y_pred, y_prob


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — HGB analysis
# ══════════════════════════════════════════════════════════════════════════════

def hgb_evaluate(df_train, df_test, specs):
    y_train, y_test = df_train["label"].values, df_test["label"].values
    rows, roc_curves, fitted = [], [], {}

    for name, cfg in specs.items():
        X_tr = df_train[cfg["features"]].values
        X_te = df_test[cfg["features"]].values
        mdl  = clone(cfg["model"])
        mdl.fit(X_tr, y_train)
        y_pred = mdl.predict(X_te)
        y_prob = mdl.predict_proba(X_te)[:, 1]

        m = compute_metrics(y_test, y_pred, y_prob)
        m["model"]      = name
        m["n_features"] = len(cfg["features"])
        rows.append(m)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves.append((name, fpr, tpr))
        fitted[name] = {"model": mdl, "y_pred": y_pred, "y_prob": y_prob}

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(HGB_OUT / "temporal_test_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for name, fpr, tpr in roc_curves:
        curve_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} ({curve_auc:.4f})")
        axes[1].plot(fpr, tpr, linewidth=2, label=name)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    axes[0].set_title("Temporal Holdout ROC Curves")
    axes[1].set_xlim(0, 0.05); axes[1].set_ylim(0.95, 1.0)
    axes[1].set_title("ROC Zoom (Top-Left)")
    plt.tight_layout()
    plt.savefig(HGB_OUT / "temporal_holdout_roc.png", dpi=170)
    plt.close()

    return metrics_df, fitted


def hgb_distribution_shift(df_train, df_test):
    rows = []
    for feat in ORIGINAL_FEATURES:
        tr, te = df_train[feat].values, df_test[feat].values
        smd = (np.mean(te) - np.mean(tr)) / (np.std(tr) + 1e-9)
        rows.append({
            "feature":    feat,
            "train_skew": float(skew(tr, nan_policy="omit")),
            "test_skew":  float(skew(te, nan_policy="omit")),
            "abs_smd":    float(abs(smd)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(HGB_OUT / "distribution_shift_checks.csv", index=False)
    return out


def hgb_diagnostics(name, bundle, df_test, specs):
    model  = bundle["model"]
    y_true = df_test["label"].values
    y_pred = bundle["y_pred"]

    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    cm_df = pd.DataFrame(cm, index=["true_not_cloud", "true_cloud"],
                         columns=["pred_not_cloud", "pred_cloud"])
    cm_df.to_csv(HGB_OUT / f"{name}_confusion_matrix.csv")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(HGB_OUT / f"{name}_confusion_matrix.png", dpi=170)
    plt.close()

    feats = specs[name]["features"]
    idx   = np.random.default_rng(42).choice(len(df_test), size=min(25000, len(df_test)), replace=False)
    p_imp = permutation_importance(
        model, df_test.iloc[idx][feats].values, df_test.iloc[idx]["label"].values,
        n_repeats=6, random_state=42, n_jobs=-1, scoring="roc_auc",
    )
    imp_df = pd.DataFrame({
        "feature": feats,
        "mean": p_imp.importances_mean,
        "std":  p_imp.importances_std,
    }).sort_values("mean", ascending=False)
    imp_df.to_csv(HGB_OUT / f"{name}_permutation_importance.csv", index=False)
    plt.figure(figsize=(8, 6))
    top = imp_df.head(15).iloc[::-1]
    plt.barh(top["feature"], top["mean"])
    plt.xlabel("Permutation Importance (mean ΔAUC)")
    plt.title(f"Top Feature Importances — {name}")
    plt.tight_layout()
    plt.savefig(HGB_OUT / f"{name}_perm_importance_top15.png", dpi=170)
    plt.close()

    if hasattr(model, "train_score_") and hasattr(model, "validation_score_"):
        its = np.arange(1, len(model.train_score_) + 1)
        pd.DataFrame({"iter": its, "train_loss": model.train_score_,
                      "val_loss": model.validation_score_}).to_csv(
            HGB_OUT / f"{name}_convergence.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.plot(its, model.train_score_, label="Train loss", linewidth=2)
        plt.plot(its, model.validation_score_, label="Val loss", linewidth=2)
        plt.xlabel("Boosting iteration"); plt.ylabel("Loss")
        plt.title(f"Convergence — {name}"); plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(HGB_OUT / f"{name}_convergence.png", dpi=170)
        plt.close()


def hgb_posthoc_error(name, bundle, df_test):
    out = df_test[["image", "x", "y", "label"]].copy()
    out["pred"]       = bundle["y_pred"]
    out["prob_cloud"] = bundle["y_prob"]
    out["is_error"]   = (out["pred"] != out["label"]).astype(int)
    out["error_type"] = np.select(
        [(out["label"] == 1) & (out["pred"] == -1),
         (out["label"] == -1) & (out["pred"] == 1)],
        ["FN_cloud_missed", "FP_false_cloud"], default="correct")
    out.to_csv(HGB_OUT / f"{name}_test_predictions.csv", index=False)

    plot_spatial_errors(
        out["x"],
        out["y"],
        out["label"],
        out["pred"],
        f"{name} — Spatial Errors (Test)",
        HGB_OUT / f"{name}_spatial_error_map.png",
    )

    quant_rows = []
    for feat in ["SD", "CORR", "DF", "AF", "AN", "NDAI_DF_AF"]:
        grp = out.groupby(pd.qcut(df_test[feat], q=10, duplicates="drop"),
                          observed=True)["is_error"].agg(["mean", "count"]).reset_index()
        grp.columns = ["bin", "error_rate", "n"]; grp["feature"] = feat
        quant_rows.append(grp)
    pd.concat(quant_rows, ignore_index=True).to_csv(
        HGB_OUT / f"{name}_error_by_feature_quantile.csv", index=False)


def hgb_boundary_error(name, bundle, df_test):
    out = df_test[["x", "y", "label"]].copy()
    out["pred"]     = bundle["y_pred"]
    out["is_error"] = (out["pred"] != out["label"]).astype(int)

    tree_cloud = KDTree(out[out["label"] ==  1][["x", "y"]].values)
    tree_clear = KDTree(out[out["label"] == -1][["x", "y"]].values)
    dist = np.zeros(len(out))
    m_cl = out["label"] == -1; m_cd = out["label"] == 1
    dist[m_cl], _ = tree_cloud.query(out.loc[m_cl, ["x", "y"]].values)
    dist[m_cd], _ = tree_clear.query(out.loc[m_cd, ["x", "y"]].values)
    out["dist_to_boundary"] = dist
    out["dist_bin"] = pd.cut(dist, bins=[0, 1.5, 3, 6, 12, 25, 50, 100, 500])

    stats = out.groupby("dist_bin", observed=True).agg(
        error_rate=("is_error", "mean"), count=("is_error", "count")).reset_index()
    stats.to_csv(HGB_OUT / f"{name}_boundary_error_stats.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=stats, x="dist_bin", y="error_rate", color="skyblue")
    plt.title(f"Error Rate vs. Distance to Label Boundary — {name}")
    plt.ylabel("Mean Error Rate"); plt.xlabel("Distance (units)")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(HGB_OUT / f"{name}_boundary_error_rate.png", dpi=170)
    plt.close()

    feat_rows = []
    for feat in ["SD", "CORR"]:
        out[feat] = df_test[feat].values
        feat_rows.append({
            "feature":   feat,
            "mean_near": out[out["dist_to_boundary"] < 15][feat].mean(),
            "mean_far":  out[out["dist_to_boundary"] > 100][feat].mean(),
        })
    pd.DataFrame(feat_rows).to_csv(HGB_OUT / f"{name}_boundary_feature_shifts.csv", index=False)


def hgb_error_groups(name, bundle, df_test):
    out = df_test.copy()
    out["pred"] = bundle["y_pred"]
    out["group"] = np.select(
        [(out["label"] == 1) & (out["pred"] == 1),
         (out["label"] == -1) & (out["pred"] == -1),
         (out["label"] == -1) & (out["pred"] == 1),
         (out["label"] == 1) & (out["pred"] == -1)],
        ["TP", "TN", "FP_false_cloud", "FN_missed_cloud"], default="unknown")

    rows = []
    for feat in ORIGINAL_FEATURES:
        g = out.groupby("group", observed=True)[feat].agg(["mean", "std", "median"]).reset_index()
        g["feature"] = feat; rows.append(g)
    pd.concat(rows, ignore_index=True).to_csv(HGB_OUT / f"{name}_error_group_stats.csv", index=False)

    plt.figure(figsize=(15, 5))
    for i, feat in enumerate(["SD", "CORR", "NDAI_DF_AF"], 1):
        plt.subplot(1, 3, i)
        sns.violinplot(data=out, x="group", y=feat,
                       order=["TN", "TP", "FP_false_cloud", "FN_missed_cloud"])
        plt.title(f"{feat} by Group"); plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(HGB_OUT / f"{name}_error_group_violins.png", dpi=170)
    plt.close()


def hgb_predict_unlabeled(specs, df_train, model_name="hgb_original"):
    feats = specs[model_name]["features"]
    mdl   = clone(specs[model_name]["model"])
    mdl.fit(df_train[feats].values, df_train["label"].values)

    candidates = sorted([p for p in DATA_DIR.glob("O*.npz") if p.stem not in LABELED_IDS])[:3]
    rows = []
    for path in candidates:
        npz  = np.load(path)
        arr  = npz[list(npz.files)[0]]
        cols = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
        if arr.shape[1] == 11:
            cols.append("label")
        df = pd.DataFrame(arr, columns=cols)
        df["NDAI_DF_AF"] = (df["DF"] - df["AF"]) / (df["DF"] + df["AF"] + 1e-9)
        prob = mdl.predict_proba(df[feats].values)[:, 1]
        pred = np.where(prob >= 0.5, 1, -1)
        rows.append({"image": path.stem, "n_pixels": len(df),
                     "pred_cloud_rate": float((pred == 1).mean()),
                     "mean_cloud_prob": float(prob.mean())})
        vis = df[["x", "y"]].copy()
        vis["prob_cloud"] = prob
        vis = vis.sample(n=min(60000, len(vis)), random_state=42)
        plt.figure(figsize=(7, 5.5))
        sc = plt.scatter(vis["x"], vis["y"], c=vis["prob_cloud"], s=1.2, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, label="P(cloud)")
        plt.title(f"Unlabeled sanity — {path.stem} ({model_name})")
        plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
        plt.savefig(HGB_OUT / f"unlabeled_{path.stem}_{model_name}_probmap.png", dpi=170)
        plt.close()

    pd.DataFrame(rows).to_csv(HGB_OUT / f"unlabeled_sanity_{model_name}.csv", index=False)


def hgb_write_summary(metrics_df, shift_df, selected):
    sel  = metrics_df[metrics_df["model"] == selected].iloc[0]
    base = metrics_df[metrics_df["model"] == "rf_base"].iloc[0]
    top3 = shift_df.sort_values("abs_smd", ascending=False).head(3)

    lines = [
        "# Classifier Research Summary", "",
        "## Candidate Classifiers",
        "- rf_base: Literature RF (SD, CORR, NDAI)",
        "- rf_full: Full RF (base + AE features)",
        "- hgb_original: HGB (base engineered features)", "",
        "## Model Assumption Checks",
        "- Tree ensembles: nonparametric, no normality/linearity assumptions.",
        "- Temporal stability checked via SMD between training and holdout.",
        "- Largest |SMD|:",
    ]
    for _, r in top3.iterrows():
        lines.append(f"  - {r['feature']}: |SMD|={r['abs_smd']:.3f}, "
                     f"train skew={r['train_skew']:.3f}, test skew={r['test_skew']:.3f}")
    lines += ["", "## Fit Assessment (Temporal Holdout: O013490)"]
    for _, r in metrics_df.iterrows():
        lines.append(f"- {r['model']}: ROC AUC={r['roc_auc']:.4f}, F1={r['f1']:.4f}")
    lines += ["", "## Selected Classifier",
              f"- Selected: {selected} (ROC AUC={sel['roc_auc']:.4f}, F1={sel['f1']:.4f}).",
              f"- Baseline rf_base: ROC AUC={base['roc_auc']:.4f}, F1={base['f1']:.4f}."]
    (HGB_OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — RF analysis
# ══════════════════════════════════════════════════════════════════════════════

def rf_baseline(model, df_train, df_test):
    print("\n--- RF Baseline ---")
    for df, split in [(df_train, "Train"), (df_test, "Test")]:
        X, y   = df[FULL_FEATURES], df["label"]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        m = compute_metrics(y, y_pred, y_prob)
        print(f"  {split}: AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

    pd.DataFrame([
        {"split": "train", **compute_metrics(df_train["label"],
                                             model.predict(df_train[FULL_FEATURES]),
                                             model.predict_proba(df_train[FULL_FEATURES])[:, 1])},
        {"split": "test",  **compute_metrics(df_test["label"],
                                             model.predict(df_test[FULL_FEATURES]),
                                             model.predict_proba(df_test[FULL_FEATURES])[:, 1])},
    ]).to_csv(RF_OUT / "rf_train_test_comparison.csv", index=False)

    rf_train_pred = model.predict(df_train[FULL_FEATURES])
    rf_test_pred = model.predict(df_test[FULL_FEATURES])
    plot_spatial_errors(
        df_train["x"],
        df_train["y"],
        df_train["label"],
        rf_train_pred,
        "RF — Spatial Errors (Train)",
        RF_OUT / "rf_train_spatial_errors.png",
    )
    plot_spatial_errors(
        df_test["x"],
        df_test["y"],
        df_test["label"],
        rf_test_pred,
        "RF — Spatial Errors (Test)",
        RF_OUT / "rf_test_spatial_errors.png",
    )

    for split, df in [("Train", df_train), ("Test", df_test)]:
        y, y_pred = df["label"], model.predict(df[FULL_FEATURES])
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix — {split}"); plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(RF_OUT / f"rf_{split.lower()}_cm.png", dpi=300); plt.close()


def rf_unlabeled(model, df_test):
    print("\n--- RF Unlabeled Pixels ---")
    df_unlab   = df_test[df_test["label"] == 0].copy()
    df_labeled = df_test[df_test["label"] != 0].copy()

    df_unlab["rf_pred"]         = model.predict(df_unlab[FULL_FEATURES])
    df_unlab["rf_prob_cloud"]   = model.predict_proba(df_unlab[FULL_FEATURES])[:, 1]
    df_labeled["rf_pred"]       = model.predict(df_labeled[FULL_FEATURES])
    df_labeled["rf_prob_cloud"] = model.predict_proba(df_labeled[FULL_FEATURES])[:, 1]

    counts = df_unlab["rf_pred"].value_counts()
    print(f"  Unlabeled pixels: {len(df_unlab):,}")
    print(f"    Cloud:     {counts.get(1, 0):,} ({counts.get(1,0)/len(df_unlab)*100:.1f}%)")
    print(f"    Non-cloud: {counts.get(-1,0):,} ({counts.get(-1,0)/len(df_unlab)*100:.1f}%)")
    acc     = accuracy_score(df_labeled["label"], df_labeled["rf_pred"])
    auc_val = roc_auc_score(df_labeled["label"], df_labeled["rf_prob_cloud"])
    print(f"  Labeled sanity check: Acc={acc:.4f}  AUC={auc_val:.4f}")

    df_unlab[["image", "y", "x", "label", "rf_pred", "rf_prob_cloud"]].to_csv(
        RF_OUT / "rf_unlabeled_predictions.csv", index=False)

    # Spatial map: unlabeled only
    sample = df_unlab.sample(n=min(30000, len(df_unlab)), random_state=42)
    plt.figure(figsize=(10, 8))
    for val, color, lbl in [(1, "#1f77b4", "Pred: Cloud"), (-1, "#d62728", "Pred: Non-cloud")]:
        sub = sample[sample["rf_pred"] == val]
        plt.scatter(sub["x"], sub["y"], c=color, s=2, alpha=0.5, label=lbl)
    plt.title("RF Predictions — Unlabeled Pixels")
    plt.legend(markerscale=5); plt.tight_layout()
    plt.savefig(RF_OUT / "rf_unlabeled_spatial_map.png", dpi=300); plt.close()

    # Combined spatial map: labeled + unlabeled (same rendering path as logreg/svm)
    all_pred = model.predict(df_test[FULL_FEATURES])
    plot_labeled_vs_unlabeled(
        df_test["x"],
        df_test["y"],
        all_pred,
        df_test["label"],
        "RF Predictions — Labeled vs. Unlabeled",
        RF_OUT / "rf_unlabeled_vs_labeled_spatial.png",
    )

    # Probability histograms
    bins = np.linspace(0, 1, 51)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                          gridspec_kw={"hspace": 0.08})
    for ax, series, color, title in [
        (ax_top, df_labeled["rf_prob_cloud"], "#e07b39",
         f"Labeled pixels (n={len(df_labeled):,})"),
        (ax_bot, df_unlab["rf_prob_cloud"],   "#3a78b5",
         f"Unlabeled pixels (n={len(df_unlab):,})"),
    ]:
        ax.hist(series, bins=bins, color=color, alpha=0.45, edgecolor="white", density=True)
        kde = gaussian_kde(series, bw_method=0.05)
        xs  = np.linspace(0, 1, 300)
        ax.plot(xs, kde(xs), color=color, linewidth=2)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Decision boundary")
        ax.set_ylabel("Density"); ax.set_title(title, fontsize=10, pad=4)
        ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.25)
    ax_bot.set_xlabel("P(cloud)")
    fig.suptitle("RF Cloud Probability Distribution", fontsize=12, y=1.01)
    plt.savefig(RF_OUT / "rf_unlabeled_prob_hist.png", dpi=300, bbox_inches="tight"); plt.close()


def rf_label_flip(df_train, df_test, n_repeats=RF_FLIP_REPS):
    print(f"\n--- RF Label Flip (rate={RF_FLIP_RATE}, reps={n_repeats}) ---")
    X_tr = df_train[FULL_FEATURES].values
    y_tr = df_train["label"].values
    X_te = df_test[FULL_FEATURES].values
    y_te = df_test["label"].values

    rf_clean = RandomForestClassifier(random_state=42, **RF_FLIP_PARAMS)
    rf_clean.fit(X_tr, y_tr)
    p_clean  = rf_clean.predict(X_te)
    pb_clean = rf_clean.predict_proba(X_te)[:, 1]
    baseline = {"condition": "clean", "flipped": 0,
                "auc": roc_auc_score(y_te, pb_clean),
                "f1":  f1_score(y_te, p_clean, pos_label=1),
                "acc": accuracy_score(y_te, p_clean)}
    print(f"  clean | AUC={baseline['auc']:.4f}  F1={baseline['f1']:.4f}  Acc={baseline['acc']:.4f}")

    flip_records = []
    for rep in range(n_repeats):
        rng     = np.random.default_rng(rep * 17)
        y_noisy = y_tr.copy()
        mask    = rng.random(len(y_noisy)) < RF_FLIP_RATE
        y_noisy[mask] *= -1
        rf = RandomForestClassifier(random_state=rep, **RF_FLIP_PARAMS)
        rf.fit(X_tr, y_noisy)
        p  = rf.predict(X_te);  pb = rf.predict_proba(X_te)[:, 1]
        rec = {"condition": f"flip_rep{rep}", "flipped": int(mask.sum()),
               "auc": roc_auc_score(y_te, pb),
               "f1":  f1_score(y_te, p, pos_label=1),
               "acc": accuracy_score(y_te, p)}
        flip_records.append(rec)
        print(f"  rep{rep} | {rec['flipped']} flipped | AUC={rec['auc']:.4f}  "
              f"F1={rec['f1']:.4f}  Acc={rec['acc']:.4f}")

    aucs = [r["auc"] for r in flip_records]
    f1s  = [r["f1"]  for r in flip_records]
    accs = [r["acc"] for r in flip_records]
    print(f"  mean±std: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
          f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    df_res = pd.DataFrame([baseline] + flip_records)
    df_res.to_csv(RF_OUT / "rf_label_flip_stability.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, col, label in zip(axes, ["auc", "f1", "acc"],
                               ["ROC AUC", "F1 Score", "Accuracy"]):
        flip_mean = np.mean([r[col] for r in flip_records])
        flip_std  = np.std([r[col]  for r in flip_records])
        ax.bar(["Clean", f"{int(RF_FLIP_RATE*100)}% flipped"],
               [baseline[col], flip_mean], color=["#2ca02c", "#d62728"], alpha=0.75, width=0.5)
        ax.errorbar(1, flip_mean, yerr=flip_std, fmt="none", color="black", capsize=6)
        lo = min(baseline[col], flip_mean - flip_std) * 0.995
        hi = max(baseline[col], flip_mean + flip_std) * 1.002
        ax.set_ylim(lo, hi); ax.set_ylabel(label); ax.set_title(label); ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"RF Test Performance: Clean vs. {int(RF_FLIP_RATE*100)}% Label Flip "
                 f"({n_repeats} repeats)", y=1.02)
    plt.tight_layout()
    plt.savefig(RF_OUT / "rf_label_flip_stability.png", dpi=300, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sns.set_theme(style="whitegrid")
    np.random.seed(42)

    df_train, df_test = load_data()
    df_train_labeled = df_train[df_train["label"] != 0].copy()
    df_test_labeled = df_test[df_test["label"] != 0].copy()

    print(f"Using labeled train rows for supervision: {len(df_train_labeled):,}")
    print(f"Using labeled test rows for supervision:  {len(df_test_labeled):,}")

    # ── Train ──
    print("\n=== Training ===")
    X_train = df_train_labeled[TRAIN_FEATURES]
    y_train = df_train_labeled["label"]

    rf  = train_rf(X_train, y_train)
    evaluate_and_save(rf, df_test_labeled, "rf")

    hgb = train_hgb(X_train, y_train)
    evaluate_and_save(hgb, df_test_labeled, "hgb")

    # ── HGB analysis ──
    print("\n=== HGB Comparative Analysis ===")
    specs = model_specs()

    print("1) Temporal holdout evaluation ...")
    metrics_df, fitted = hgb_evaluate(df_train_labeled, df_test_labeled, specs)

    print("2) Distribution shift check ...")
    shift_df = hgb_distribution_shift(df_train_labeled, df_test_labeled)

    selected = "hgb_original"
    print(f"3) Diagnostics for {selected} ...")
    hgb_diagnostics(selected, fitted[selected], df_test_labeled, specs)

    print("4) Post-hoc error EDA ...")
    hgb_posthoc_error(selected, fitted[selected], df_test_labeled)
    hgb_boundary_error(selected, fitted[selected], df_test_labeled)
    hgb_error_groups(selected, fitted[selected], df_test_labeled)

    print("5) Unlabeled image sanity predictions ...")
    hgb_predict_unlabeled(specs, df_train_labeled)

    print("6) Writing summary ...")
    hgb_write_summary(metrics_df, shift_df, selected)

    # ── RF analysis ──
    print("\n=== RF Analysis ===")

    print("7) RF baseline evaluation ...")
    rf_baseline(rf, df_train_labeled, df_test_labeled)

    print("8) RF unlabeled pixel predictions ...")
    rf_unlabeled(rf, df_test)

    print("9) RF label flip stability ...")
    rf_label_flip(df_train_labeled, df_test_labeled)

    print(f"\nDone.\n  Models  → {RESULTS_DIR}\n"
          f"  HGB     → {HGB_OUT}\n  RF      → {RF_OUT}")


if __name__ == "__main__":
    main()
