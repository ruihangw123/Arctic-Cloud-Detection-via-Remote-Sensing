"""
logreg_svm_stability.py — Label-flip stability check for LogReg and SVM.

Mutates 5% of training labels (flip sign) then reruns the full SVM + LogReg
pipeline across multiple seeds to verify model robustness to label noise.

Outputs saved to report/figures/model/
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Allow direct execution and module import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logreg_svm import (  # noqa: E402
    FEATURE_COLS,
    SVM_COLS,
    load_data,
    plot_spatial_errors,
    plot_labeled_vs_unlabeled,
    plot_svm_diagnostics,
    plot_logreg_diagnostics,
    plot_prob_histogram,
    run_stepwise_selection,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "report" / "figures" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLIP_RATE = 0.05
FLIP_REPS = 3  # number of random seeds for the flip experiment


# ── Label mutation ─────────────────────────────────────────────────────────────

def flip_labels(y: pd.Series, rate: float, seed: int) -> pd.Series:
    """Randomly flip `rate` fraction of labels from ±1 to ∓1."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    idx = rng.choice(len(y_noisy), size=int(rate * len(y_noisy)), replace=False)
    y_noisy.iloc[idx] = -y_noisy.iloc[idx]
    print(f"  Flipped {len(idx)}/{len(y_noisy)} labels ({rate*100:.0f}%)")
    return y_noisy


# ── Shared stability bar-chart (matches RF label-flip style) ───────────────────

def plot_stability_bars(baseline: dict, flip_records: list[dict],
                        model_name: str, out_path: Path):
    """
    3-panel bar chart (AUC / F1 / Accuracy) matching the ensemble RF style:
    green bar = clean, red bar = flipped (mean ± std).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, col, label in zip(axes, ["auc", "f1", "acc"],
                               ["ROC AUC", "F1 Score", "Accuracy"]):
        flip_mean = np.mean([r[col] for r in flip_records])
        flip_std  = np.std([r[col]  for r in flip_records])
        ax.bar(["Clean", f"{int(FLIP_RATE*100)}% flipped"],
               [baseline[col], flip_mean],
               color=["#2ca02c", "#d62728"], alpha=0.75, width=0.5)
        ax.errorbar(1, flip_mean, yerr=flip_std,
                    fmt="none", color="black", capsize=6)
        lo = min(baseline[col], flip_mean - flip_std) * 0.995
        hi = max(baseline[col], flip_mean + flip_std) * 1.002
        ax.set_ylim(lo, hi)
        ax.set_ylabel(label); ax.set_title(label); ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{model_name} Test Performance: Clean vs. "
                 f"{int(FLIP_RATE*100)}% Label Flip ({FLIP_REPS} repeats)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── SVM stability ──────────────────────────────────────────────────────────────

def run_svm_stability(df_train, df_val, df_test, df_opt):
    X_train = df_train[SVM_COLS]
    X_test  = df_test[SVM_COLS];  y_test = df_test["label"]

    # Clean baseline
    svm_clean = SVC(kernel="rbf", gamma=1.0, C=1.0)
    svm_clean.fit(X_train, df_train["label"])
    p_clean = svm_clean.predict(X_test)
    baseline = {
        "condition": "clean",
        "auc": roc_auc_score(y_test, p_clean),
        "f1":  roc_auc_score(y_test, p_clean),   # SVC has no predict_proba by default
        "acc": accuracy_score(y_test, p_clean),
    }
    # Use decision_function for AUC
    baseline["auc"] = roc_auc_score(y_test, svm_clean.decision_function(X_test))

    print(f"  clean | AUC={baseline['auc']:.4f}  Acc={baseline['acc']:.4f}")

    flip_records = []
    for rep in range(FLIP_REPS):
        y_noisy = flip_labels(df_train["label"], FLIP_RATE, seed=rep)
        svm = SVC(kernel="rbf", gamma=1.0, C=1.0)
        svm.fit(X_train, y_noisy)
        p    = svm.predict(X_test)
        auc_ = roc_auc_score(y_test, svm.decision_function(X_test))
        rec  = {"condition": f"flip_rep{rep}",
                "auc": auc_,
                "f1":  roc_auc_score(y_test, p),
                "acc": accuracy_score(y_test, p)}
        flip_records.append(rec)
        print(f"  rep{rep} | AUC={rec['auc']:.4f}  Acc={rec['acc']:.4f}")

    plot_stability_bars(baseline, flip_records, "SVM",
                        OUT_DIR / "fig_model_stab_svm_label_flip.png")

    # Spatial and diagnostic plots for the last flip rep
    svm_last = SVC(kernel="rbf", gamma=1.0, C=1.0)
    svm_last.fit(X_train, flip_labels(df_train["label"], FLIP_RATE, seed=0))

    plot_svm_diagnostics(svm_last, X_test, y_test,
                         f"Test Set ({int(FLIP_RATE*100)}% Label Flip)",
                         OUT_DIR / "fig_model_stab_svm_diagnostics.png")

    plot_spatial_errors(df_test["x"], df_test["y"], y_test,
                        svm_last.predict(X_test),
                        f"SVM Stability — Spatial Errors ({int(FLIP_RATE*100)}% flip)",
                        OUT_DIR / "fig_model_stab_svm_spatial_errors.png")

    plot_labeled_vs_unlabeled(df_opt["x"], df_opt["y"],
                              svm_last.predict(df_opt[SVM_COLS]), df_opt["label"],
                              f"SVM Stability — Labeled vs. Unlabeled ({int(FLIP_RATE*100)}% flip)",
                              OUT_DIR / "fig_model_stab_svm_labeled_vs_unlabeled.png")

    print(f"  mean±std: AUC={np.mean([r['auc'] for r in flip_records]):.4f}"
          f"±{np.std([r['auc'] for r in flip_records]):.4f}")


# ── LogReg stability ───────────────────────────────────────────────────────────

def run_logreg_stability(df_train, df_val, df_test, df_opt):
    X_train = df_train[FEATURE_COLS]
    X_test  = df_test[FEATURE_COLS];  y_test = df_test["label"]

    # Clean baseline — fit sfs and lr on clean labels
    sfs_clean, _ = run_stepwise_selection(X_train, df_train["label"])
    lr_clean = LogisticRegression(solver="liblinear", class_weight="balanced")
    lr_clean.fit(sfs_clean.transform(X_train), df_train["label"])
    p_clean   = lr_clean.predict(sfs_clean.transform(X_test))
    pb_clean  = lr_clean.predict_proba(sfs_clean.transform(X_test))[:, 1]
    baseline  = {
        "condition": "clean",
        "auc": roc_auc_score(y_test, pb_clean),
        "f1":  roc_auc_score(y_test, p_clean),
        "acc": accuracy_score(y_test, p_clean),
    }
    baseline["f1"] = f1_score(y_test, p_clean, pos_label=1)
    print(f"  clean | AUC={baseline['auc']:.4f}  F1={baseline['f1']:.4f}  Acc={baseline['acc']:.4f}")

    flip_records = []
    for rep in range(FLIP_REPS):
        y_noisy = flip_labels(df_train["label"], FLIP_RATE, seed=rep)
        sfs, feature_names = run_stepwise_selection(X_train, y_noisy)
        lr = LogisticRegression(solver="liblinear", class_weight="balanced")
        lr.fit(sfs.transform(X_train), y_noisy)
        p  = lr.predict(sfs.transform(X_test))
        pb = lr.predict_proba(sfs.transform(X_test))[:, 1]
        rec = {"condition": f"flip_rep{rep}",
               "auc": roc_auc_score(y_test, pb),
               "f1":  f1_score(y_test, p, pos_label=1),
               "acc": accuracy_score(y_test, p)}
        flip_records.append(rec)
        print(f"  rep{rep} | AUC={rec['auc']:.4f}  F1={rec['f1']:.4f}  Acc={rec['acc']:.4f}")

    plot_stability_bars(baseline, flip_records, "LogReg",
                        OUT_DIR / "fig_model_stab_logreg_label_flip.png")

    # Spatial and diagnostic plots for the last flip rep
    y_noisy_0      = flip_labels(df_train["label"], FLIP_RATE, seed=0)
    sfs_f, fn_f    = run_stepwise_selection(X_train, y_noisy_0)
    lr_f           = LogisticRegression(solver="liblinear", class_weight="balanced")
    lr_f.fit(sfs_f.transform(X_train), y_noisy_0)
    test_preds     = lr_f.predict(sfs_f.transform(X_test))
    test_prob      = lr_f.predict_proba(sfs_f.transform(X_test))[:, 1]

    plot_logreg_diagnostics(lr_f, sfs_f, X_test, y_test, fn_f)

    plot_prob_histogram(test_prob, y_test,
                        f"LogReg Stability — Probabilities (Test, {int(FLIP_RATE*100)}% flip)",
                        OUT_DIR / "fig_model_stab_logreg_prob_hist_test.png")

    df_unlab   = df_opt[df_opt["label"] == 0]
    prob_unlab = lr_f.predict_proba(sfs_f.transform(df_unlab[FEATURE_COLS]))[:, 1]
    pred_unlab = lr_f.predict(sfs_f.transform(df_unlab[FEATURE_COLS]))
    plot_prob_histogram(prob_unlab, pred_unlab,
                        f"LogReg Stability — Probabilities (Unlabeled, {int(FLIP_RATE*100)}% flip)",
                        OUT_DIR / "fig_model_stab_logreg_prob_hist_unlabeled.png")

    plot_spatial_errors(df_test["x"], df_test["y"], y_test, test_preds,
                        f"LogReg Stability — Spatial Errors ({int(FLIP_RATE*100)}% flip)",
                        OUT_DIR / "fig_model_stab_logreg_spatial_errors.png")

    all_pred = lr_f.predict(sfs_f.transform(df_opt[FEATURE_COLS]))
    plot_labeled_vs_unlabeled(df_opt["x"], df_opt["y"], all_pred, df_opt["label"],
                              f"LogReg Stability — Labeled vs. Unlabeled ({int(FLIP_RATE*100)}% flip)",
                              OUT_DIR / "fig_model_stab_logreg_labeled_vs_unlabeled.png")

    print(f"  mean±std: AUC={np.mean([r['auc'] for r in flip_records]):.4f}"
          f"±{np.std([r['auc'] for r in flip_records]):.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sns.set_theme(style="whitegrid")
    np.random.seed(42)

    df_train, df_val, df_test, df_opt = load_data()

    print("\n=== SVM Label-Flip Stability ===")
    run_svm_stability(df_train, df_val, df_test, df_opt)

    print("\n=== LogReg Label-Flip Stability ===")
    run_logreg_stability(df_train, df_val, df_test, df_opt)

    print(f"\nDone. Figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
