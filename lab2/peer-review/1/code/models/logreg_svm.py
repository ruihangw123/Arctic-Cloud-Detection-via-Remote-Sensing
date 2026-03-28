"""
logreg_svm.py — Logistic Regression and SVM models for cloud classification.

Sections:
  1. SVM (RBF kernel, feature-selected columns)
  2. Logistic Regression (Lasso L1 + stepwise forward selection)

Outputs saved to results/logreg_svm/
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "feature_eng_dataset"
OUT_DIR     = ROOT / "results" / "logreg_svm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_COLS = ["SD", "CORR", "DF", "CF", "BF", "AF", "AN", "NDAI_DF_AF", "PC1"]
SVM_COLS     = ["SD", "CORR", "NDAI_DF_AF"]

# Shared spatial-map style (aligned with ensemble.py)
SPATIAL_COLORS = {
    (False, True):  "#b37a4c",  # labeled   + pred cloud
    (False, False): "#5f7f63",  # labeled   + pred non-cloud
    (True,  True):  "#4f6f8f",  # unlabeled + pred cloud
    (True,  False): "#8a5a5a",  # unlabeled + pred non-cloud
}
SPATIAL_FIGSIZE = (11, 8)
SPATIAL_DPI = 220
SPATIAL_MARKER = "s"
SPATIAL_POINT_SIZE = 2.5
SPATIAL_ALPHA = 0.55
SPATIAL_MAX_PER_GROUP = 40000


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    df_all   = pd.read_csv(DATASET_DIR / "train_features_opt.csv")
    df_opt   = pd.read_csv(DATASET_DIR / "test_features_opt.csv")
    df_test  = df_opt[df_opt["label"] != 0].copy()

    df_train = df_all[df_all["image"] == "O013257"].drop(columns=["image"])
    df_val   = df_all[df_all["image"] == "O012791"].drop(columns=["image"])

    print(f"Train : {df_train.shape}  Val : {df_val.shape}  Test : {df_test.shape}")
    print(f"Opt (labeled+unlabeled): {df_opt.shape}")
    return df_train, df_val, df_test, df_opt


# ── Shared plot helpers ────────────────────────────────────────────────────────

def plot_spatial_errors(x_coords, y_coords, y_true, y_pred, title, out_path):
    """Spatial error map for labeled test data."""
    colors = np.empty(len(y_true), dtype=object)
    colors[y_true == y_pred]                       = "lightgrey"  # correct
    colors[(y_true == -1) & (y_pred == 1)]         = "red"        # FP
    colors[(y_true == 1)  & (y_pred == -1)]        = "darkred"    # FN

    legend_elements = [
        mlines.Line2D([0], [0], marker="s", color="w", label="Correct",                   markerfacecolor="lightgrey", markersize=10),
        mlines.Line2D([0], [0], marker="s", color="w", label="False Positive (FP)",       markerfacecolor="red",       markersize=10),
        mlines.Line2D([0], [0], marker="s", color="w", label="False Negative (FN)",       markerfacecolor="darkred",   markersize=10),
    ]

    plt.figure(figsize=(12, 10))
    plt.scatter(x_coords, y_coords, c=colors, s=1, marker="s")
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(title)
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def plot_labeled_vs_unlabeled(x_coords, y_coords, y_pred, real_labels, title, out_path):
    """Spatial map showing labeled vs. unlabeled pixels with shared color palette."""
    plot_df = pd.DataFrame({
        "x": x_coords,
        "y": y_coords,
        "pred": y_pred,
        "is_unlabeled": (real_labels == 0),
    })

    x_lim = (plot_df["x"].min(), plot_df["x"].max())
    y_lim = (plot_df["y"].min(), plot_df["y"].max())

    layers = [
        (False, False, "Labeled: Pred Non-cloud"),
        (False, True,  "Labeled: Pred Cloud"),
        (True,  False, "Unlabeled: Pred Non-cloud"),
        (True,  True,  "Unlabeled: Pred Cloud"),
    ]

    plt.figure(figsize=SPATIAL_FIGSIZE)
    for is_unlab, is_cloud, label in layers:
        pred_val = 1 if is_cloud else -1
        sub = plot_df[(plot_df["is_unlabeled"] == is_unlab) & (plot_df["pred"] == pred_val)]
        if len(sub) > SPATIAL_MAX_PER_GROUP:
            sub = sub.sample(SPATIAL_MAX_PER_GROUP, random_state=42)
        plt.scatter(
            sub["x"],
            sub["y"],
            c=SPATIAL_COLORS[(is_unlab, is_cloud)],
            s=SPATIAL_POINT_SIZE,
            marker=SPATIAL_MARKER,
            alpha=SPATIAL_ALPHA,
            linewidths=0,
            label=label,
        )

    plt.xlim(x_lim); plt.ylim(y_lim)
    plt.legend(loc="upper right", frameon=True, markerscale=6)
    plt.title(title)
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=SPATIAL_DPI); plt.close()


# ── Section 1: SVM ─────────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    print("Training SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", gamma=1.0, C=1.0)
    svm.fit(X_train, y_train)
    return svm


def plot_svm_diagnostics(svm_model, X, y, title_suffix, out_path):
    """4-panel SVM diagnostic plot: PCA clusters, SV density, margin dist, feature dist."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    y_pred = svm_model.predict(X)
    decision_values = svm_model.decision_function(X)
    sv_indices = np.abs(decision_values) <= 1.0

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # 1) PCA clusters
    axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="coolwarm", s=10, alpha=0.6)
    axes[0, 0].set_title("Decision Clusters in PCA Space (2D Projection)")
    axes[0, 0].set_xlabel("Principal Component 1"); axes[0, 0].set_ylabel("Principal Component 2")

    # 2) Support vector density
    sns.kdeplot(x=X_pca[sv_indices, 0], y=X_pca[sv_indices, 1],
                fill=True, cmap="Reds", ax=axes[0, 1])
    axes[0, 1].set_title("Support Vector Density")
    axes[0, 1].set_xlabel("PC1"); axes[0, 1].set_ylabel("PC2")

    # 3) Margin distribution
    sns.histplot(decision_values[y == 1],  color="red",  label="Cloud (Actual)", kde=True, ax=axes[1, 0])
    sns.histplot(decision_values[y == -1], color="blue", label="Clear (Actual)", kde=True, ax=axes[1, 0])
    axes[1, 0].axvline(0, color="black", linestyle="--")
    axes[1, 0].set_title("Distance to SVM Hyperplane (Margin Distribution)")
    axes[1, 0].set_xlabel("Functional Distance"); axes[1, 0].legend()

    # 4) Feature distribution by class
    X_plot = pd.DataFrame(X, columns=SVM_COLS)
    X_plot["label"] = y.values if hasattr(y, "values") else y
    df_melted = pd.melt(X_plot, id_vars="label", var_name="Feature", value_name="Value")
    sns.boxplot(data=df_melted, x="Value", y="Feature", hue="label",
                palette={1: "red", -1: "blue"}, ax=axes[1, 1], fliersize=0)
    handles, _ = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles, ["Clear (-1)", "Cloud (1)"], loc="lower right")
    axes[1, 1].set_title("Feature Distribution by Class")
    axes[1, 1].set_xlabel("Scaled Feature Value"); axes[1, 1].set_xlim(-3, 3)

    plt.suptitle(f"SVM Diagnostics — {title_suffix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def run_svm(df_train, df_val, df_test, df_opt):
    X_train = df_train[SVM_COLS]; y_train = df_train["label"]
    X_val   = df_val[SVM_COLS];   y_val   = df_val["label"]
    X_test  = df_test[SVM_COLS];  y_test  = df_test["label"]

    svm = train_svm(X_train, y_train)

    val_preds  = svm.predict(X_val)
    test_preds = svm.predict(X_test)

    print("\n--- SVM Results ---")
    print(f"Val  Accuracy : {accuracy_score(y_val, val_preds):.4f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_preds):.4f}")
    print(f"Test ROC AUC  : {roc_auc_score(y_test, test_preds):.4f}")
    print(classification_report(y_test, test_preds))

    plot_svm_diagnostics(svm, X_test, y_test, "Test Set", OUT_DIR / "svm_diagnostics_test.png")

    df_unlab = df_opt[df_opt["label"] == 0]
    X_unlab  = df_unlab[SVM_COLS]
    y_unlab_pred = svm.predict(X_unlab)
    plot_svm_diagnostics(svm, X_unlab, y_unlab_pred, "Unlabeled (Self-Predicted)",
                         OUT_DIR / "svm_diagnostics_unlabeled.png")

    plot_spatial_errors(df_test["x"], df_test["y"], y_test, test_preds,
                        "SVM — Spatial Errors (Test)", OUT_DIR / "svm_spatial_errors_test.png")

    plot_labeled_vs_unlabeled(df_opt["x"], df_opt["y"],
                              svm.predict(df_opt[SVM_COLS]), df_opt["label"],
                              "SVM Predictions — Labeled vs. Unlabeled",
                              OUT_DIR / "svm_labeled_vs_unlabeled.png")

    return svm


# ── Section 2: Logistic Regression ────────────────────────────────────────────

def run_lasso_selection(X_train_scaled, y_train, X_train):
    print("\nPerforming variable selection (Lasso L1)...")
    selector_model = LogisticRegression(penalty="l1", C=0.01, solver="liblinear", random_state=42)
    selector = SelectFromModel(selector_model)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    selected_features = X_train.columns[selector.get_support()]
    print(f"  Features selected: {len(selected_features)} — {list(selected_features)}")

    selector_model.fit(X_train_selected, y_train)
    return selector_model, selector, selected_features


def run_stepwise_selection(X_train, y_train):
    print(f"\nStepwise forward selection on {X_train.shape[1]} features...")
    lr_base = LogisticRegression(solver="liblinear", class_weight="balanced")
    sfs = SequentialFeatureSelector(lr_base, n_features_to_select=5,
                                    direction="forward", scoring="accuracy", cv=3, n_jobs=-1)
    sfs.fit(X_train, y_train)
    feature_names = X_train.columns[sfs.get_support()]
    print(f"  Selected: {list(feature_names)}")
    return sfs, feature_names


def plot_logreg_diagnostics(lr, sfs, X_test, y_test, feature_names):
    X_te = sfs.transform(X_test)
    y_pred = lr.predict(X_te)
    y_prob = lr.predict_proba(X_te)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted Label"); axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xticklabels(["Clear (-1)", "Cloud (1)"])
    axes[0, 0].set_yticklabels(["Clear (-1)", "Cloud (1)"])

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_val:.2f}")
    axes[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0, 1].set_title("ROC Curve"); axes[0, 1].set_xlabel("FPR"); axes[0, 1].set_ylabel("TPR")
    axes[0, 1].legend(loc="lower right")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    axes[1, 0].plot(recall, precision, color="green", lw=2, label=f"Avg Precision = {ap_score:.2f}")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].set_xlabel("Recall"); axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend(loc="lower left")

    # Feature coefficients
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": lr.coef_[0]}).sort_values("Coefficient")
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="viridis", ax=axes[1, 1])
    axes[1, 1].set_title("Feature Importance (LogReg Coefficients)")
    axes[1, 1].axvline(0, color="black", lw=1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "logreg_diagnostics.png", dpi=150); plt.close()


def plot_prob_histogram(y_prob, y_labels, title, out_path):
    """Stacked probability histogram for cloud vs. clear."""
    plt.figure(figsize=(8, 5))
    sns.histplot(y_prob[y_labels == 1],  color="red",  label="Cloud",  kde=True, stat="density", alpha=0.5)
    sns.histplot(y_prob[y_labels == -1], color="blue", label="Clear",  kde=True, stat="density", alpha=0.5)
    plt.title(title)
    plt.xlabel("Probability of being Cloud (Class 1)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def run_logreg(df_train, df_val, df_test, df_opt):
    X_train = df_train[FEATURE_COLS]; y_train = df_train["label"]
    X_val   = df_val[FEATURE_COLS];   y_val   = df_val["label"]
    X_test  = df_test[FEATURE_COLS];  y_test  = df_test["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    run_lasso_selection(X_train_scaled, y_train, X_train)

    sfs, feature_names = run_stepwise_selection(X_train, y_train)

    lr = LogisticRegression(solver="liblinear", class_weight="balanced")
    lr.fit(sfs.transform(X_train), y_train)

    val_preds  = lr.predict(sfs.transform(X_val))
    test_preds = lr.predict(sfs.transform(X_test))
    test_prob  = lr.predict_proba(sfs.transform(X_test))[:, 1]

    print("\n--- LogReg Results ---")
    print(f"Val  Accuracy : {accuracy_score(y_val, val_preds):.4f}")
    print(f"Test Accuracy : {accuracy_score(y_test, test_preds):.4f}")
    print(f"Test ROC AUC  : {roc_auc_score(y_test, test_prob):.4f}")
    print(classification_report(y_test, test_preds))

    plot_logreg_diagnostics(lr, sfs, X_test, y_test, feature_names)

    # Probability histograms — labeled test
    plot_prob_histogram(test_prob, y_test,
                        "LogReg — Predicted Probabilities (Test)",
                        OUT_DIR / "logreg_prob_hist_test.png")

    # Probability histograms — unlabeled
    df_unlab    = df_opt[df_opt["label"] == 0]
    X_unlab     = sfs.transform(df_unlab[FEATURE_COLS])
    prob_unlab  = lr.predict_proba(X_unlab)[:, 1]
    pred_unlab  = lr.predict(X_unlab)
    plot_prob_histogram(prob_unlab, pred_unlab,
                        "LogReg — Predicted Probabilities (Unlabeled)",
                        OUT_DIR / "logreg_prob_hist_unlabeled.png")

    # Spatial error map — test
    plot_spatial_errors(df_test["x"], df_test["y"], y_test, test_preds,
                        "LogReg — Spatial Errors (Test)", OUT_DIR / "logreg_spatial_errors_test.png")

    # Labeled vs. unlabeled spatial map
    all_pred = lr.predict(sfs.transform(df_opt[FEATURE_COLS]))
    plot_labeled_vs_unlabeled(df_opt["x"], df_opt["y"], all_pred, df_opt["label"],
                              "LogReg Predictions — Labeled vs. Unlabeled",
                              OUT_DIR / "logreg_labeled_vs_unlabeled.png")

    return lr, sfs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sns.set_theme(style="whitegrid")

    df_train, df_val, df_test, df_opt = load_data()

    print("\n=== SVM ===")
    run_svm(df_train, df_val, df_test, df_opt)

    print("\n=== Logistic Regression ===")
    run_logreg(df_train, df_val, df_test, df_opt)

    print(f"\nDone. Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
