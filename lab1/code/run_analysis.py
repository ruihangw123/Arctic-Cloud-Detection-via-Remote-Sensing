"""
Run full Lab 1 analysis: load data, clean, fit models, save figures.
Call from project root: python code/run_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ensure code directory is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean import load_and_clean
from models import (
    predict_pecarn,
    build_logistic_model,
    predict_logistic,
    build_random_forest,
    predict_random_forest,
)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "TBI PUD 10-08-2013.csv"
    figures_dir = PROJECT_ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    print("Loading and cleaning data...")
    df = load_and_clean(str(data_path))
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Filter to analysis cohort: GCS 14-15, non-missing outcome (as in Kuppermann)
    # Outcome: PosCT (positive CT) or DeathTBI/ciTBI - use PosCT for CT recommendation target
    df_analysis = df[
        (df["GCSTotal"].between(14, 15))
        & (df["PosCT"].notna())
    ].copy()
    df_analysis["y"] = (df_analysis["PosCT"] == 1).astype(int)
    print(f"Analysis cohort (GCS 14-15, non-missing PosCT): {len(df_analysis)} rows")

    # PECARN CDR predictions
    y_pecarn = predict_pecarn(df_analysis)

    acc_pecarn = accuracy_score(df_analysis["y"], y_pecarn)
    print(f"PECARN CDR accuracy (on full cohort): {acc_pecarn:.4f}")

    # Logistic regression: use subset of PECARN-relevant features
    feat_cols = [
        "GCSTotal", "AMS", "SFxPalp", "Vomit", "LOCSeparate", "High_impact_InjSev",
        "HASeverity", "Hema", "HemaLoc", "ActNorm", "LocLen",
        "AgeTwoPlus", "AgeinYears",
    ]
    available = [c for c in feat_cols if c in df_analysis.columns]
    X = df_analysis[available].fillna(0)  # Simple imputation for modeling
    y = df_analysis["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    lr_model, lr_scaler = build_logistic_model(X_train, y_train)
    y_lr = predict_logistic(lr_model, lr_scaler, X_test)
    acc_lr = accuracy_score(y_test, y_lr)
    print(f"Logistic regression accuracy: {acc_lr:.4f}")

    rf_model = build_random_forest(X_train, y_train)
    y_rf = predict_random_forest(rf_model, X_test)
    acc_rf = accuracy_score(y_test, y_rf)
    print(f"Random forest accuracy: {acc_rf:.4f}")

    # Example figure: age distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    df_analysis["AgeinYears"].dropna().hist(ax=ax, bins=20, edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age distribution in analysis cohort")
    fig.tight_layout()
    fig.savefig(figures_dir / "age_distribution.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'age_distribution.pdf'}")

    # Example figure: PECARN rule vs outcome
    fig, ax = plt.subplots(figsize=(5, 4))
    cross_tab = pd.crosstab(y_pecarn, df_analysis["y"], normalize="index")
    cross_tab.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_xlabel("PECARN: CT recommended")
    ax.set_ylabel("Proportion")
    ax.set_title("PECARN rule recommendation vs. positive CT outcome")
    ax.legend(["No PosCT", "PosCT"])
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "pecarn_vs_outcome.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'pecarn_vs_outcome.pdf'}")

    # Stability check: before (all) vs after (exclude youngest, AgeInMonth<24)
    ct_done = df_analysis[df_analysis["CTDone"] == 1]
    ct_perturbed = ct_done[ct_done["AgeInMonth"].notna() & (ct_done["AgeInMonth"] >= 24)]

    sev_labels = {1.0: "Low", 2.0: "Moderate", 3.0: "High"}
    before_sev = ct_done.groupby("High_impact_InjSev")["y"].mean()
    after_sev = ct_perturbed.groupby("High_impact_InjSev")["y"].mean()
    before_sev.index = [sev_labels.get(i, str(i)) for i in before_sev.index]
    after_sev.index = [sev_labels.get(i, str(i)) for i in after_sev.index]

    # Line plot: two lines (before vs after) - shows comparison without duplicating bar style
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(3)
    ax.plot(x, before_sev.values * 100, "o-", label="All patients", linewidth=2, markersize=8)
    ax.plot(x, after_sev.values * 100, "s--", label="Exclude age < 2 y", linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Moderate", "High"])
    ax.set_xlabel("Injury mechanism severity")
    ax.set_ylabel("Positive CT rate (%)")
    ax.set_title("Stability: injury-severity gradient before vs after perturbation")
    ax.legend()
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(figures_dir / "stability_check.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'stability_check.pdf'}")

    # Model comparison
    fig, ax = plt.subplots(figsize=(5, 4))
    models = ["PECARN CDR", "Logistic Regression", "Random Forest"]
    accs = [acc_pecarn, acc_lr, acc_rf]
    bars = ax.bar(models, accs, color=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model comparison (test set)")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.2f}", ha="center", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(figures_dir / "model_comparison.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'model_comparison.pdf'}")

    # Logistic regression coefficients
    coef_df = pd.DataFrame({
        "feature": available,
        "coefficient": lr_model.coef_[0],
    })
    coef_df = coef_df.loc[coef_df["coefficient"].abs().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_df["coefficient"]]
    ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Coefficient")
    ax.set_title("Logistic regression coefficients")
    plt.tight_layout()
    fig.savefig(figures_dir / "lr_coefficients.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'lr_coefficients.pdf'}")

    # Random forest variable importance
    imp = pd.DataFrame({
        "feature": available,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp["feature"], imp["importance"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Random forest variable importance")
    plt.tight_layout()
    fig.savefig(figures_dir / "rf_importance.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'rf_importance.pdf'}")

    # Scatter: LR coefficient vs RF importance - shows model alignment (different graph type)
    lr_coef = pd.Series(lr_model.coef_[0], index=available)
    rf_imp = pd.Series(rf_model.feature_importances_, index=available)
    df_align = pd.DataFrame({"LR_coef": lr_coef, "RF_imp": rf_imp})
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_align["LR_coef"], df_align["RF_imp"], alpha=0.7)
    for idx in df_align.index:
        ax.annotate(idx, (df_align.loc[idx, "LR_coef"], df_align.loc[idx, "RF_imp"]),
                    fontsize=8, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Logistic regression coefficient")
    ax.set_ylabel("Random forest importance")
    ax.set_title("Model alignment: LR vs RF predictor emphasis")
    plt.tight_layout()
    fig.savefig(figures_dir / "model_alignment.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'model_alignment.pdf'}")


if __name__ == "__main__":
    main()
