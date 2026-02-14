"""
Run full Lab 1 analysis: load data, clean, fit models, save figures.
Call from project root: python code/run_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split

# Ensure code directory is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean import load_and_clean
from labels import get_label
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

    # ================================================================
    # Sensitivity / Specificity / PPV / NPV analysis
    # ================================================================
    from sklearn.metrics import recall_score, precision_score

    # PECARN on full analysis cohort
    y_full = df_analysis["y"].values
    tn, fp, fn, tp = confusion_matrix(y_full, y_pecarn).ravel()
    pecarn_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    pecarn_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    pecarn_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    pecarn_npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # LR on test set
    tn, fp, fn, tp = confusion_matrix(y_test, y_lr).ravel()
    lr_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    lr_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    lr_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    lr_npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # RF on test set
    tn, fp, fn, tp = confusion_matrix(y_test, y_rf).ravel()
    rf_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    rf_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    rf_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    rf_npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"PECARN: Sens={pecarn_sens:.3f}, Spec={pecarn_spec:.3f}, PPV={pecarn_ppv:.3f}, NPV={pecarn_npv:.3f}")
    print(f"LR:     Sens={lr_sens:.3f}, Spec={lr_spec:.3f}, PPV={lr_ppv:.3f}, NPV={lr_npv:.3f}")
    print(f"RF:     Sens={rf_sens:.3f}, Spec={rf_spec:.3f}, PPV={rf_ppv:.3f}, NPV={rf_npv:.3f}")

    # Grouped bar chart: sensitivity, specificity, PPV, NPV for each model
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["Sensitivity", "Specificity", "PPV", "NPV"]
    x = np.arange(len(metrics))
    w = 0.25
    pecarn_vals = [pecarn_sens, pecarn_spec, pecarn_ppv, pecarn_npv]
    lr_vals = [lr_sens, lr_spec, lr_ppv, lr_npv]
    rf_vals = [rf_sens, rf_spec, rf_ppv, rf_npv]
    bars1 = ax.bar(x - w, pecarn_vals, w, label="PECARN CDR", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x, lr_vals, w, label="Logistic Regression", color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x + w, rf_vals, w, label="Random Forest", color="#e74c3c", alpha=0.85)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Diagnostic performance: Sensitivity, Specificity, PPV, NPV")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(figures_dir / "diagnostic_performance.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'diagnostic_performance.pdf'}")

    # ================================================================
    # Selection bias: CT vs non-CT patient characteristics
    # ================================================================
    df_broad = df[(df["GCSTotal"].between(14, 15))].copy()
    df_broad["ct_group"] = np.where(df_broad["CTDone"] == 1, "CT done", "No CT")
    compare_vars = ["AgeinYears", "AMS", "Vomit", "LOCSeparate", "High_impact_InjSev"]
    avail_compare = [v for v in compare_vars if v in df_broad.columns]

    fig, axes = plt.subplots(1, len(avail_compare), figsize=(14, 3.5))
    for ax, var in zip(axes, avail_compare):
        ct_vals = df_broad[df_broad["ct_group"] == "CT done"][var].dropna()
        noct_vals = df_broad[df_broad["ct_group"] == "No CT"][var].dropna()
        if var == "AgeinYears":
            # Overlaid histograms for continuous variable
            ax.hist(noct_vals, bins=18, alpha=0.5, density=True, label="No CT", color="#3498db", edgecolor="white")
            ax.hist(ct_vals, bins=18, alpha=0.5, density=True, label="CT done", color="#e74c3c", edgecolor="white")
            ax.set_ylabel("Density")
        else:
            # Mean comparison for binary/ordinal variables
            means = [noct_vals.mean(), ct_vals.mean()]
            ax.bar(["No CT", "CT done"], means, color=["#3498db", "#e74c3c"], alpha=0.8)
            ax.set_ylabel("Mean value")
        ax.set_title(get_label(var), fontsize=10)
        if var == "AgeinYears":
            ax.legend(fontsize=8)
    fig.suptitle("Selection bias: characteristics of CT vs non-CT patients", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "selection_bias.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {figures_dir / 'selection_bias.pdf'}")

    # ================================================================
    # GCS 14 vs 15 subgroup analysis
    # ================================================================
    gcs14 = df_analysis[df_analysis["GCSTotal"] == 14]
    gcs15 = df_analysis[df_analysis["GCSTotal"] == 15]
    gcs14_rate = gcs14["y"].mean() * 100
    gcs15_rate = gcs15["y"].mean() * 100
    print(f"GCS 14: PosCT={gcs14_rate:.1f}% (n={len(gcs14)}), GCS 15: PosCT={gcs15_rate:.1f}% (n={len(gcs15)})")

    # Compare risk factor prevalence
    rf_compare = {}
    for rf in ["AMS", "Vomit", "LOCSeparate", "High_impact_InjSev"]:
        if rf in df_analysis.columns:
            # For High_impact_InjSev: compare % high-impact (=3)
            if rf == "High_impact_InjSev":
                rf_compare[get_label(rf)] = [
                    (gcs14[rf] == 3).mean() * 100,
                    (gcs15[rf] == 3).mean() * 100,
                ]
            else:
                rf_compare[get_label(rf)] = [
                    (gcs14[rf] == 1).mean() * 100,
                    (gcs15[rf] == 1).mean() * 100,
                ]
    rf_df = pd.DataFrame(rf_compare, index=["GCS 14", "GCS 15"]).T

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: PosCT rate comparison
    axes[0].bar(["GCS 14", "GCS 15"], [gcs14_rate, gcs15_rate],
                color=["#e74c3c", "#3498db"], alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Positive CT rate (%)")
    axes[0].set_title("PosCT rate by GCS score")
    for i, v in enumerate([gcs14_rate, gcs15_rate]):
        axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)
    # Right: Risk factor prevalence
    x = np.arange(len(rf_df))
    w = 0.35
    axes[1].barh(x - w / 2, rf_df["GCS 14"], w, label="GCS 14", color="#e74c3c", alpha=0.85)
    axes[1].barh(x + w / 2, rf_df["GCS 15"], w, label="GCS 15", color="#3498db", alpha=0.85)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(rf_df.index, fontsize=9)
    axes[1].set_xlabel("Prevalence (%)")
    axes[1].set_title("Risk factor prevalence by GCS")
    axes[1].legend(fontsize=9)
    fig.suptitle("GCS 14 vs 15: outcome and risk factor profile", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "gcs14_vs_15.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {figures_dir / 'gcs14_vs_15.pdf'}")

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

    # ================================================================
    # DEEPER EXPLORATION: Relationships between variables
    # ================================================================

    # --- 1. Risk factor accumulation: PosCT rate by number of concurrent risk factors ---
    risk_factor_cols = ["AMS", "Vomit", "LOCSeparate", "SFxPalp", "Hema", "SFxBas"]
    df_analysis["n_risk_factors"] = 0
    for rf in risk_factor_cols:
        if rf in df_analysis.columns:
            df_analysis["n_risk_factors"] += (df_analysis[rf] == 1).fillna(False).astype(int)
    # High-impact injury mechanism as an additional factor
    if "High_impact_InjSev" in df_analysis.columns:
        df_analysis["n_risk_factors"] += (df_analysis["High_impact_InjSev"] == 3).fillna(False).astype(int)
    # GCS = 14 as a factor
    df_analysis["n_risk_factors"] += (df_analysis["GCSTotal"] == 14).fillna(False).astype(int)

    rf_rates = df_analysis.groupby("n_risk_factors")["y"].agg(["mean", "count"])
    rf_rates = rf_rates[rf_rates["count"] >= 10]
    # Cap at 4+ for display
    df_analysis["n_risk_capped"] = df_analysis["n_risk_factors"].clip(upper=4)
    rf_capped = df_analysis.groupby("n_risk_capped")["y"].agg(["mean", "count"])
    rf_capped.index = [str(int(i)) if i < 4 else "4+" for i in rf_capped.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rf_capped.index, rf_capped["mean"] * 100,
                  color="steelblue", alpha=0.85, edgecolor="white")
    for bar, (idx, row) in zip(bars, rf_capped.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={int(row['count'])}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Number of PECARN risk factors present")
    ax.set_ylabel("Positive CT rate (%)")
    ax.set_title("Compound risk: PosCT rate by number of concurrent risk factors")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_factor_accumulation.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'risk_factor_accumulation.pdf'}")

    # --- 2. Age-stratified predictor effects ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pred_specs = [
        ("High_impact_InjSev", "Injury Severity", {1.0: "Low", 2.0: "Mod", 3.0: "High"}),
        ("AMS", "Altered Mental Status", {0.0: "No", 1.0: "Yes"}),
        ("Vomit", "Vomiting", {0.0: "No", 1.0: "Yes"}),
    ]
    age_styles = [(1, "<2 years", "#e74c3c", "o-"), (2, "\u22652 years", "#3498db", "s--")]
    for ax, (pred, label, tick_map) in zip(axes, pred_specs):
        for age_grp, age_label, color, style in age_styles:
            sub = df_analysis[df_analysis["AgeTwoPlus"] == age_grp]
            rates = sub.groupby(pred)["y"].agg(["mean", "count"])
            rates = rates[rates["count"] >= 20]
            x_vals = rates.index.values
            ax.plot(range(len(x_vals)), rates["mean"].values * 100, style,
                    label=age_label, color=color, linewidth=2, markersize=8)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels([tick_map.get(v, str(v)) for v in x_vals])
        ax.set_xlabel(label)
        ax.set_ylabel("Positive CT rate (%)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, None)
    fig.suptitle("Age-stratified predictor effects on Positive CT rate", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "age_stratified_effects.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {figures_dir / 'age_stratified_effects.pdf'}")

    # --- 3. Interaction: AMS x Injury Severity -> PosCT rate ---
    sev_labels = {1.0: "Low", 2.0: "Moderate", 3.0: "High"}
    ams_labels = {0.0: "No AMS", 1.0: "AMS present"}
    interact = df_analysis.groupby(["AMS", "High_impact_InjSev"])["y"].agg(["mean", "count"])
    interact = interact[interact["count"] >= 10].reset_index()
    interact["sev_label"] = interact["High_impact_InjSev"].map(sev_labels)
    interact["ams_label"] = interact["AMS"].map(ams_labels)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(3)
    width = 0.35
    for i, (ams_val, ams_label, color) in enumerate(
        [(0.0, "No AMS", "#3498db"), (1.0, "AMS present", "#e74c3c")]
    ):
        sub = interact[interact["AMS"] == ams_val].sort_values("High_impact_InjSev")
        if len(sub) == 3:
            bars = ax.bar(x_pos + i * width, sub["mean"] * 100, width,
                          label=ams_label, color=color, alpha=0.85, edgecolor="white")
            for bar, (_, row) in zip(bars, sub.iterrows()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{row['mean']*100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(["Low", "Moderate", "High"])
    ax.set_xlabel("Injury Mechanism Severity")
    ax.set_ylabel("Positive CT rate (%)")
    ax.set_title("Interaction: AMS \u00d7 Injury Severity on Positive CT Rate")
    ax.legend()
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(figures_dir / "interaction_ams_severity.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'interaction_ams_severity.pdf'}")

    # --- 4. CT utilization vs actual PosCT by risk factor count ---
    # Among the broader cohort (GCS 14-15), compute CT scan rate and PosCT rate by risk count
    df_broad = df[
        (df["GCSTotal"].between(14, 15))
    ].copy()
    df_broad["n_risk_factors"] = 0
    for rf in risk_factor_cols:
        if rf in df_broad.columns:
            df_broad["n_risk_factors"] += (df_broad[rf] == 1).fillna(False).astype(int)
    if "High_impact_InjSev" in df_broad.columns:
        df_broad["n_risk_factors"] += (df_broad["High_impact_InjSev"] == 3).fillna(False).astype(int)
    df_broad["n_risk_factors"] += (df_broad["GCSTotal"] == 14).fillna(False).astype(int)
    df_broad["n_risk_capped"] = df_broad["n_risk_factors"].clip(upper=4)
    df_broad["ct_done"] = (df_broad["CTDone"] == 1).fillna(False).astype(int)

    ct_util = df_broad.groupby("n_risk_capped").agg(
        ct_rate=("ct_done", "mean"),
        n=("ct_done", "count"),
    )
    ct_util.index = [str(int(i)) if i < 4 else "4+" for i in ct_util.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ct_util.index, ct_util["ct_rate"] * 100,
           color="#9b59b6", alpha=0.85, edgecolor="white")
    for i, (idx, row) in enumerate(ct_util.iterrows()):
        ax.text(i, row["ct_rate"] * 100 + 1,
                f"n={int(row['n'])}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("Number of PECARN risk factors present")
    ax.set_ylabel("CT scan rate (%)")
    ax.set_title("CT utilization by number of risk factors (all GCS 14-15 patients)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(figures_dir / "ct_utilization_by_risk.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'ct_utilization_by_risk.pdf'}")

    # --- 5. Vomiting x LOC interaction ---
    vomit_loc = df_analysis.groupby(["Vomit", "LOCSeparate"])["y"].agg(["mean", "count"])
    vomit_loc = vomit_loc[vomit_loc["count"] >= 10].reset_index()
    # Simplify LOC: 0=No, 1/2=Yes/Suspected -> group
    vomit_loc["LOC_any"] = (vomit_loc["LOCSeparate"] >= 1).astype(int)
    vomit_loc_agg = df_analysis.copy()
    vomit_loc_agg["LOC_any"] = (vomit_loc_agg["LOCSeparate"] >= 1).fillna(False).astype(int)
    interaction2 = vomit_loc_agg.groupby(["Vomit", "LOC_any"])["y"].agg(["mean", "count"])
    interaction2 = interaction2[interaction2["count"] >= 10].reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels_combo = []
    vals = []
    colors_combo = []
    combo_map = {
        (0.0, 0): ("Neither", "#2ecc71"),
        (1.0, 0): ("Vomit only", "#f39c12"),
        (0.0, 1): ("LOC only", "#3498db"),
        (1.0, 1): ("Both", "#e74c3c"),
    }
    for (v, l), (lbl, clr) in combo_map.items():
        row = interaction2[(interaction2["Vomit"] == v) & (interaction2["LOC_any"] == l)]
        if len(row) == 1:
            labels_combo.append(lbl)
            vals.append(row["mean"].values[0] * 100)
            colors_combo.append(clr)
    bars = ax.bar(labels_combo, vals, color=colors_combo, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Positive CT rate (%)")
    ax.set_title("Vomiting \u00d7 LOC Interaction: Compound Effect on PosCT")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(figures_dir / "interaction_vomit_loc.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'interaction_vomit_loc.pdf'}")

    # ================================================================
    # END DEEPER EXPLORATION
    # ================================================================

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
    coef_df = coef_df.copy()
    coef_df["label"] = [get_label(f) for f in coef_df["feature"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_df["coefficient"]]
    ax.barh(coef_df["label"], coef_df["coefficient"], color=colors, alpha=0.8)
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
    imp = imp.copy()
    imp["label"] = [get_label(f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp["label"], imp["importance"], color="steelblue", alpha=0.8)
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
    df_align["label"] = [get_label(f) for f in df_align.index]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df_align["LR_coef"], df_align["RF_imp"], alpha=0.7)
    for idx in df_align.index:
        ax.annotate(df_align.loc[idx, "label"],
                    (df_align.loc[idx, "LR_coef"], df_align.loc[idx, "RF_imp"]),
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
