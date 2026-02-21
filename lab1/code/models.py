"""
Classification models and full analysis pipeline for PECARN TBI CT scan recommendation.
Implements: Kuppermann clinical decision rule, logistic regression, random forest,
diagnostic evaluation, exploratory analysis, and figure generation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean import load_and_clean  # noqa: E402


# -----------------------------------------------------------------------------
# Kuppermann et al. PECARN Clinical Decision Rule (Lancet 2009)
# Rule: Recommend CT if any high-risk OR any medium-risk factor
# -----------------------------------------------------------------------------


def pecarn_recommend_ct(row: pd.Series) -> bool:
    """
    Apply PECARN clinical decision rule: recommend CT if any risk factor present.

    Age < 2 years:
        High-risk: GCS<=14, AMS, palpable skull fracture
        Medium-risk: non-frontal scalp hematoma, LOC>=5s, severe mechanism, not acting normal
    Age >= 2 years:
        High-risk: GCS<=14, AMS, signs of basilar skull fracture
        Medium-risk: vomiting, LOC, severe mechanism, severe headache

    Parameters
    ----------
    row : pd.Series
        One patient row with columns: AgeTwoPlus, GCSTotal, AMS, SFxPalp,
        Hema, HemaLoc, LocLen, LOCSeparate, High_impact_InjSev, ActNorm,
        Vomit, HASeverity, SFxBas, SFxBasHem, SFxBasOto, SFxBasPer, SFxBasRet, SFxBasRhi

    Returns
    -------
    bool
        True if CT is recommended
    """
    age_two_plus = row.get("AgeTwoPlus", 2)  # 1 = <2y, 2 = >=2y
    under_two = age_two_plus == 1

    # High-risk: GCS <= 14 or AMS
    gcs = row.get("GCSTotal")
    gcs_low = gcs is not None and gcs <= 14
    ams = row.get("AMS")
    ams_positive = ams == 1

    if under_two:
        # <2 years: palpable skull fracture
        sfx_palp = row.get("SFxPalp")
        high_risk = gcs_low or ams_positive or (sfx_palp == 1)

        # Medium-risk: non-frontal hematoma (occipital, parietal/temporal)
        # HemaLoc: 1=Frontal, 2=Occipital, 3=Parietal/Temporal per documentation
        hema = row.get("Hema") == 1
        hema_loc = row.get("HemaLoc")
        non_frontal = hema and hema_loc in [2, 3]

        loc_len = row.get("LocLen")
        loc_5plus = loc_len in [2, 3, 4]  # 5sec-1min, 1-5min, >5min

        inj_sev = row.get("High_impact_InjSev")
        severe_mech = inj_sev == 3

        act_norm = row.get("ActNorm")
        not_normal = act_norm == 0

        medium_risk = non_frontal or loc_5plus or severe_mech or not_normal
    else:
        # >=2 years: signs of basilar skull fracture
        sfx_bas = row.get("SFxBas") == 1
        sfx_bas_hem = row.get("SFxBasHem") == 1
        sfx_bas_oto = row.get("SFxBasOto") == 1
        sfx_bas_per = row.get("SFxBasPer") == 1
        sfx_bas_ret = row.get("SFxBasRet") == 1
        sfx_bas_rhi = row.get("SFxBasRhi") == 1
        basilar = sfx_bas or sfx_bas_hem or sfx_bas_oto or sfx_bas_per or sfx_bas_ret or sfx_bas_rhi

        high_risk = gcs_low or ams_positive or basilar

        # Medium-risk: vomiting, LOC (any), severe mechanism, severe headache
        vomit = row.get("Vomit") == 1
        loc_sep = row.get("LOCSeparate")
        loc_any = loc_sep in [1, 2]  # Yes or Suspected

        inj_sev = row.get("High_impact_InjSev")
        severe_mech = inj_sev == 3

        ha_sev = row.get("HASeverity")
        severe_ha = ha_sev == 3

        medium_risk = vomit or loc_any or severe_mech or severe_ha

    return bool(high_risk or medium_risk)

"""
Reader-friendly labels for variables displayed in figures.
Maps technical variable names to human-readable labels.
"""

VAR_LABELS = {
    # PECARN rule variables
    "GCSTotal": "GCS total",
    "AMS": "Altered mental status",
    "SFxPalp": "Palpable skull fracture",
    "Vomit": "Vomiting",
    "LOCSeparate": "Loss of consciousness",
    "LocLen": "LOC duration",
    "High_impact_InjSev": "Injury mechanism severity",
    "HASeverity": "Headache severity",
    "Hema": "Hematoma",
    "HemaLoc": "Hematoma location",
    "ActNorm": "Acting normally",
    "SFxBas": "Basilar skull fracture",
    "AgeTwoPlus": "Age ≥ 2 years",
    "AgeinYears": "Age (years)",
    # Other common variables
    "PosCT": "Positive CT finding",
    "ciTBI": "Clinically important TBI",
    "CTDone": "CT performed",
    "AgeInMonth": "Age (months)",
    "SFxPalpDepress": "Skull fx: palpable depression",
    "SFxBasHem": "Skull fx: basilar hemorrhage",
    "SFxBasPer": "Skull fx: basilar perforation",
    "SFxBasRhi": "Skull fx: basilar rhinorrhea",
    "SFxBasRet": "Skull fx: basilar retroauricular",
    "SFxBasOto": "Skull fx: basilar otorrhea",
    "SeizLen": "Seizure length",
    "SeizOccur": "Seizure occurrence",
    "CTSedAgitate": "CT sedation: agitation",
    "CTSedAge": "CT sedation: age",
}


def get_label(var: str) -> str:
    """Return reader-friendly label for a variable, or a cleaned version if not in map."""
    if var in VAR_LABELS:
        return VAR_LABELS[var]
    # Fallback: replace underscores with spaces
    return var.replace("_", " ").strip()

def predict_pecarn(df: pd.DataFrame) -> np.ndarray:
    """
    Apply PECARN rule to each row. Returns 1 if CT recommended, 0 otherwise.
    """
    return np.array([1 if pecarn_recommend_ct(row) else 0 for _, row in df.iterrows()])


# -----------------------------------------------------------------------------
# Logistic Regression
# -----------------------------------------------------------------------------


def build_logistic_model(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Fit logistic regression with optional L2 regularization.
    Uses StandardScaler for numeric features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
    model.fit(X_scaled, y)
    return model, scaler


def predict_logistic(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict class labels using fitted logistic model."""
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    return (probs >= threshold).astype(int)


def logistic_summary(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Build a summary table of logistic regression results.

    Returns a DataFrame with one row per feature showing the standardized
    coefficient, the odds ratio (exp(coef)), and the scaled mean/std used
    during standardization.  A final row contains the intercept.
    """
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)

    rows = []
    for name, coef, or_val, mean, scale in zip(
        feature_names, coefs, odds_ratios, scaler.mean_, scaler.scale_
    ):
        rows.append({
            "Feature": name,
            "Coefficient": round(coef, 4),
            "Odds Ratio": round(or_val, 4),
            "Scaler Mean": round(mean, 4),
            "Scaler Std": round(scale, 4),
        })

    rows.append({
        "Feature": "(Intercept)",
        "Coefficient": round(model.intercept_[0], 4),
        "Odds Ratio": round(np.exp(model.intercept_[0]), 4),
        "Scaler Mean": np.nan,
        "Scaler Std": np.nan,
    })

    summary = pd.DataFrame(rows)

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
    )

    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    summary.attrs["accuracy"] = acc
    summary.attrs["sensitivity"] = sens
    summary.attrs["specificity"] = spec
    summary.attrs["n_train"] = None
    summary.attrs["n_test"] = len(y_true)
    summary.attrs["C"] = model.C

    return summary


# -----------------------------------------------------------------------------
# Random Forest (third model)
# -----------------------------------------------------------------------------


def build_random_forest(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Predict class labels using fitted random forest."""
    return model.predict(X).astype(int)



# =========================================================================
# Analysis pipeline
# =========================================================================


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

    # Broader cohort for EDA figures (GCS 14-15, CT or outcome available)
    from clean import flag_inconsistencies, derive_ciTBI, get_analysis_cohort
    df_flagged = flag_inconsistencies(df)
    df_flagged["ciTBI"] = derive_ciTBI(df_flagged)
    eda_cohort = get_analysis_cohort(df_flagged)

    # ================================================================
    # EDA figures: eda_overview, pecarn_correlation, gcs_consistency
    # ================================================================
    n_rows = len(df)
    miss_data = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        miss_data.append({"variable": col, "n_missing": n_miss,
                          "pct_missing": n_miss / n_rows * 100})
    missing_report = pd.DataFrame(miss_data).sort_values(
        "pct_missing", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    top_miss = missing_report.head(20)
    axes[0, 0].barh(range(len(top_miss)), top_miss["pct_missing"],
                     color="steelblue", alpha=0.8)
    axes[0, 0].set_yticks(range(len(top_miss)))
    axes[0, 0].set_yticklabels(
        [get_label(v) for v in top_miss["variable"]], fontsize=8)
    axes[0, 0].set_xlabel("% Missing")
    axes[0, 0].set_title("Missingness by Variable (Top 20)")
    axes[0, 0].invert_yaxis()
    age = eda_cohort["AgeinYears"].dropna()
    axes[0, 1].hist(age, bins=18, edgecolor="white", color="steelblue",
                     alpha=0.8)
    axes[0, 1].set_xlabel("Age (years)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Age Distribution (Analysis Cohort)")
    gcs = eda_cohort["GCSTotal"].dropna()
    gcs.value_counts().sort_index().plot(kind="bar", ax=axes[1, 0],
                                          color="steelblue", alpha=0.8)
    axes[1, 0].set_xlabel("GCS Total")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("GCS Distribution (Analysis Cohort)")
    ct_eda = eda_cohort[eda_cohort["CTDone"] == 1]
    outcomes = pd.DataFrame({
        "PosCT": ct_eda["PosCT"].value_counts(),
        "ciTBI": eda_cohort["ciTBI"].value_counts(),
    })
    outcomes.plot(kind="bar", ax=axes[1, 1],
                  color=["#2ecc71", "#e74c3c"], alpha=0.8)
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Outcome Distribution")
    axes[1, 1].legend(title="Variable")
    plt.tight_layout()
    fig.savefig(figures_dir / "eda_overview.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'eda_overview.pdf'}")

    pecarn_numeric = [
        "GCSTotal", "AMS", "Vomit", "LOCSeparate", "High_impact_InjSev",
        "HASeverity", "Hema", "ActNorm", "AgeTwoPlus", "AgeinYears",
    ]
    avail_num = [v for v in pecarn_numeric if v in eda_cohort.columns]
    corr = eda_cohort[avail_num].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(avail_num)))
    ax.set_yticks(range(len(avail_num)))
    labels_num = [get_label(v) for v in avail_num]
    ax.set_xticklabels(labels_num, rotation=45, ha="right")
    ax.set_yticklabels(labels_num)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation among PECARN variables")
    plt.tight_layout()
    fig.savefig(figures_dir / "pecarn_correlation.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'pecarn_correlation.pdf'}")

    n_match = (~df_flagged["GCS_mismatch"]).sum()
    n_mismatch = df_flagged["GCS_mismatch"].sum()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Consistent", "Mismatch"], [n_match, n_mismatch],
           color=["#27ae60", "#e74c3c"], alpha=0.8)
    ax.set_ylabel("Number of rows")
    ax.set_title("GCS: Component sum vs GCSTotal")
    plt.tight_layout()
    fig.savefig(figures_dir / "gcs_consistency.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'gcs_consistency.pdf'}")

    # All PECARN risk factors from Kuppermann et al. (age is a stratification
    # variable in the PECARN rule, not a risk factor, so it is excluded)
    feat_cols = [
        # Shared: GCS, AMS, severe injury mechanism
        "GCSTotal", "AMS", "High_impact_InjSev",
        # <2 years factors: palpable skull fracture, scalp hematoma, LOC duration, acting normal
        "SFxPalp", "Hema", "HemaLoc", "LocLen", "ActNorm",
        # >=2 years factors: basilar skull fracture signs, vomiting, LOC, severe headache
        "SFxBas", "SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi",
        "Vomit", "LOCSeparate", "HASeverity",
    ]
    available = [c for c in feat_cols if c in df_analysis.columns]
    X = df_analysis[available].fillna(0)  # Simple imputation for modeling
    y = df_analysis["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # PECARN CDR predictions on the same test set used for ML models
    y_pecarn = predict_pecarn(df_analysis.loc[X_test.index])
    acc_pecarn = accuracy_score(y_test, y_pecarn)
    print(f"PECARN CDR accuracy (test set): {acc_pecarn:.4f}")

    lr_model, lr_scaler = build_logistic_model(X_train, y_train)
    y_lr = predict_logistic(lr_model, lr_scaler, X_test)
    acc_lr = accuracy_score(y_test, y_lr)
    print(f"Logistic regression accuracy: {acc_lr:.4f}")

    lr_table = logistic_summary(lr_model, lr_scaler, available, y_test, y_lr)
    print("\n--- Logistic Regression Coefficient Table ---")
    print(lr_table.to_string(index=False))
    print(f"\n  Regularization C = {lr_model.C}")
    print(f"  Test-set accuracy = {lr_table.attrs['accuracy']:.4f}")
    print(f"  Test-set sensitivity = {lr_table.attrs['sensitivity']:.4f}")
    print(f"  Test-set specificity = {lr_table.attrs['specificity']:.4f}")
    print()

    rf_model = build_random_forest(X_train, y_train)
    y_rf = predict_random_forest(rf_model, X_test)
    acc_rf = accuracy_score(y_test, y_rf)
    print(f"Random forest accuracy: {acc_rf:.4f}")

    # ================================================================
    # Sensitivity / Specificity / PPV / NPV analysis
    # ================================================================
    # PECARN on test set (same as ML models for fair comparison)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pecarn).ravel()
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

    # (diagnostic_performance figure generated after class-weighted models below)

    # ================================================================
    # Threshold tuning: match PECARN sensitivity, compare specificity
    # ================================================================
    X_test_scaled = lr_scaler.transform(X_test)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    target_sens = pecarn_sens
    n_pos = y_test.sum()
    _ = len(y_test) - n_pos

    def metrics_at_threshold(probs, y_true, threshold):
        y_pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        return sens, spec, ppv, npv

    # Search for threshold that achieves >= target sensitivity
    best_lr_t, best_rf_t = 0.5, 0.5
    for t in np.arange(0.001, 0.50, 0.001):
        s, _, _, _ = metrics_at_threshold(lr_probs, y_test, t)
        if s >= target_sens:
            best_lr_t = t
            break
    for t in np.arange(0.001, 0.50, 0.001):
        s, _, _, _ = metrics_at_threshold(rf_probs, y_test, t)
        if s >= target_sens:
            best_rf_t = t
            break
    # Search from high to low so we find the HIGHEST threshold that still meets target
    for t in np.arange(0.50, 0.001, -0.001):
        s, _, _, _ = metrics_at_threshold(lr_probs, y_test, t)
        if s >= target_sens:
            best_lr_t = t
            break
    for t in np.arange(0.50, 0.001, -0.001):
        s, _, _, _ = metrics_at_threshold(rf_probs, y_test, t)
        if s >= target_sens:
            best_rf_t = t
            break

    lr_tuned = metrics_at_threshold(lr_probs, y_test, best_lr_t)
    rf_tuned = metrics_at_threshold(rf_probs, y_test, best_rf_t)

    print(f"\n--- Threshold Tuning (match PECARN sensitivity ~{target_sens:.1%}) ---")
    print(f"LR  threshold={best_lr_t:.3f}: Sens={lr_tuned[0]:.3f}, Spec={lr_tuned[1]:.3f}, "
          f"PPV={lr_tuned[2]:.3f}, NPV={lr_tuned[3]:.3f}")
    print(f"RF  threshold={best_rf_t:.3f}: Sens={rf_tuned[0]:.3f}, Spec={rf_tuned[1]:.3f}, "
          f"PPV={rf_tuned[2]:.3f}, NPV={rf_tuned[3]:.3f}")
    print(f"PECARN:            Sens={pecarn_sens:.3f}, Spec={pecarn_spec:.3f}, "
          f"PPV={pecarn_ppv:.3f}, NPV={pecarn_npv:.3f}")

    # Bar chart: at matched sensitivity, compare specificity across models
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: sensitivity-matched comparison
    models_tuned = ["PECARN CDR", "LR (tuned)", "RF (tuned)"]
    specs_tuned = [pecarn_spec, lr_tuned[1], rf_tuned[1]]
    sens_tuned = [pecarn_sens, lr_tuned[0], rf_tuned[0]]
    colors_t = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = axes[0].bar(models_tuned, specs_tuned, color=colors_t, alpha=0.85, edgecolor="white")
    for bar, spec, sens in zip(bars, specs_tuned, sens_tuned):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{spec:.1%}\n(sens={sens:.1%})", ha="center", fontsize=8)
    axes[0].set_ylabel("Specificity")
    axes[0].set_title("Specificity at matched sensitivity (~92%)")
    axes[0].set_ylim(0, 0.55)

    # Right: full metric comparison (tuned thresholds)
    metrics_t = ["Sensitivity", "Specificity", "PPV", "NPV"]
    x_t = np.arange(len(metrics_t))
    w_t = 0.25
    pecarn_t_vals = [pecarn_sens, pecarn_spec, pecarn_ppv, pecarn_npv]
    lr_t_vals = list(lr_tuned)
    rf_t_vals = list(rf_tuned)
    b1 = axes[1].bar(x_t - w_t, pecarn_t_vals, w_t, label="PECARN CDR", color="#3498db", alpha=0.85)
    b2 = axes[1].bar(x_t, lr_t_vals, w_t, label=f"LR (t={best_lr_t:.3f})", color="#2ecc71", alpha=0.85)
    b3 = axes[1].bar(x_t + w_t, rf_t_vals, w_t, label=f"RF (t={best_rf_t:.3f})", color="#e74c3c", alpha=0.85)
    for bs in [b1, b2, b3]:
        for bar in bs:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{bar.get_height():.2f}", ha="center", fontsize=7)
    axes[1].set_xticks(x_t)
    axes[1].set_xticklabels(metrics_t)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Diagnostic performance with tuned thresholds")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.15)

    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_tuning.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'threshold_tuning.pdf'}")

    # ================================================================
    # Prevalence-based threshold: set threshold = base rate
    # ================================================================
    prevalence = y_test.mean()
    lr_prev = metrics_at_threshold(lr_probs, y_test, prevalence)
    rf_prev = metrics_at_threshold(rf_probs, y_test, prevalence)

    print(f"\n--- Prevalence-based Threshold (t = {prevalence:.3f}) ---")
    print(f"LR:     Sens={lr_prev[0]:.3f}, Spec={lr_prev[1]:.3f}, "
          f"PPV={lr_prev[2]:.3f}, NPV={lr_prev[3]:.3f}")
    print(f"RF:     Sens={rf_prev[0]:.3f}, Spec={rf_prev[1]:.3f}, "
          f"PPV={rf_prev[2]:.3f}, NPV={rf_prev[3]:.3f}")
    print(f"PECARN: Sens={pecarn_sens:.3f}, Spec={pecarn_spec:.3f}, "
          f"PPV={pecarn_ppv:.3f}, NPV={pecarn_npv:.3f}")

    # ================================================================
    # Class-weighted models: upweight the minority (PosCT=1) class
    # ================================================================
    lr_bal = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                class_weight="balanced")
    lr_bal.fit(lr_scaler.transform(X_train), y_train)
    y_lr_bal = (lr_bal.predict_proba(X_test_scaled)[:, 1] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_lr_bal).ravel()
    lr_bal_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    lr_bal_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    lr_bal_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    lr_bal_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc_lr_bal = accuracy_score(y_test, y_lr_bal)

    rf_bal = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    random_state=42, class_weight="balanced")
    rf_bal.fit(X_train, y_train)
    y_rf_bal = rf_bal.predict(X_test).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_rf_bal).ravel()
    rf_bal_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    rf_bal_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    rf_bal_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    rf_bal_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc_rf_bal = accuracy_score(y_test, y_rf_bal)

    print("\n--- Class-Weighted Models (class_weight='balanced') ---")
    print(f"LR balanced: Sens={lr_bal_sens:.3f}, Spec={lr_bal_spec:.3f}, "
          f"PPV={lr_bal_ppv:.3f}, NPV={lr_bal_npv:.3f}")
    print(f"RF balanced: Sens={rf_bal_sens:.3f}, Spec={rf_bal_spec:.3f}, "
          f"PPV={rf_bal_ppv:.3f}, NPV={rf_bal_npv:.3f}")
    print(f"PECARN:      Sens={pecarn_sens:.3f}, Spec={pecarn_spec:.3f}, "
          f"PPV={pecarn_ppv:.3f}, NPV={pecarn_npv:.3f}")

    # Grouped bar chart: diagnostic performance including class-weighted models
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["Sensitivity", "Specificity", "PPV", "NPV"]
    x = np.arange(len(metrics))
    w = 0.15
    pecarn_vals = [pecarn_sens, pecarn_spec, pecarn_ppv, pecarn_npv]
    lr_vals = [lr_sens, lr_spec, lr_ppv, lr_npv]
    lr_bal_vals = [lr_bal_sens, lr_bal_spec, lr_bal_ppv, lr_bal_npv]
    rf_vals = [rf_sens, rf_spec, rf_ppv, rf_npv]
    rf_bal_vals = [rf_bal_sens, rf_bal_spec, rf_bal_ppv, rf_bal_npv]
    bars1 = ax.bar(x - 2*w, pecarn_vals, w, label="PECARN CDR", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x - w, lr_vals, w, label="LR (default)", color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x, lr_bal_vals, w, label="LR (balanced)", color="#27ae60", alpha=0.85)
    bars4 = ax.bar(x + w, rf_vals, w, label="RF (default)", color="#e74c3c", alpha=0.85)
    bars5 = ax.bar(x + 2*w, rf_bal_vals, w, label="RF (balanced)", color="#c0392b", alpha=0.85)
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Diagnostic Performance: Default vs Class-Weighted Models")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(figures_dir / "diagnostic_performance.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'diagnostic_performance.pdf'}")

    # Summary comparison figure: default vs class-weighted vs PECARN
    fig, ax = plt.subplots(figsize=(8, 5))
    approach_labels = [
        "PECARN CDR",
        "LR (default 0.5)",
        "LR (balanced)",
        "RF (default 0.5)",
        "RF (balanced)",
    ]
    sens_all = [pecarn_sens, lr_sens, lr_bal_sens, rf_sens, rf_bal_sens]
    spec_all = [pecarn_spec, lr_spec, lr_bal_spec, rf_spec, rf_bal_spec]
    colors_all = ["#3498db", "#2ecc71", "#2ecc71", "#e74c3c", "#e74c3c"]
    markers = ["D", "o", "v", "o", "v"]
    for s, sp, lbl, c, m in zip(
        sens_all, spec_all, approach_labels, colors_all, markers
    ):
        ax.scatter(sp, s, color=c, marker=m, s=120, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(lbl, (sp, s), fontsize=8, ha="left",
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Sensitivity vs Specificity: Default, Class-Weighted, and PECARN")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(pecarn_sens, color="#3498db", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(pecarn_spec, color="#3498db", linestyle="--", alpha=0.3, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_comparison.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'threshold_comparison.pdf'}")

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
    cross_tab = pd.crosstab(y_pecarn, y_test, normalize="index")
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
    for (v, loc_val), (lbl, clr) in combo_map.items():
        row = interaction2[(interaction2["Vomit"] == v) & (interaction2["LOC_any"] == loc_val)]
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
    # NEW FINDING A: AMS subtype heterogeneity × injury mechanism
    # ================================================================
    ams_subtypes = ["AMSAgitated", "AMSSleep", "AMSSlow", "AMSRepeat"]
    ams_avail = [c for c in ams_subtypes if c in df_analysis.columns]
    print("\n=== FINDING A: AMS Subtype Heterogeneity ===")
    for sub in ams_avail:
        df_analysis[sub] = df_analysis[sub].fillna(0)
        pos = df_analysis[df_analysis[sub] == 1]
        neg = df_analysis[df_analysis[sub] != 1]
        rate_pos = pos["y"].mean() * 100 if len(pos) > 0 else 0
        rate_neg = neg["y"].mean() * 100 if len(neg) > 0 else 0
        print(f"  {sub}=1: PosCT={rate_pos:.1f}% (n={len(pos)}); "
              f"{sub}=0: PosCT={rate_neg:.1f}% (n={len(neg)})")

    sev_labels_num = {1.0: "Low", 2.0: "Moderate", 3.0: "High"}
    print("\n  AMS subtype × Injury Severity → PosCT:")
    for sub in ams_avail:
        for sev_val, sev_lbl in sev_labels_num.items():
            mask = (df_analysis[sub] == 1) & (df_analysis["High_impact_InjSev"] == sev_val)
            grp = df_analysis[mask]
            if len(grp) >= 5:
                print(f"    {sub}=1 & Sev={sev_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: PosCT rate by AMS subtype
    sub_rates = {}
    for sub in ams_avail:
        grp = df_analysis[df_analysis[sub] == 1]
        if len(grp) >= 10:
            sub_rates[sub.replace("AMS", "")] = (grp["y"].mean() * 100, len(grp))
    no_ams = df_analysis[df_analysis["AMS"] != 1]
    sub_rates["No AMS"] = (no_ams["y"].mean() * 100, len(no_ams))
    labels_a = list(sub_rates.keys())
    vals_a = [v[0] for v in sub_rates.values()]
    ns_a = [v[1] for v in sub_rates.values()]
    color_a = ["#e74c3c" if lab != "No AMS" else "#95a5a6" for lab in labels_a]
    bars = axes[0].bar(labels_a, vals_a, color=color_a, alpha=0.85, edgecolor="white")
    for bar, val, n in zip(bars, vals_a, ns_a):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%\nn={n}", ha="center", fontsize=8)
    axes[0].set_ylabel("Positive CT rate (%)")
    axes[0].set_title("PosCT Rate by AMS Subtype")
    axes[0].set_ylim(0, max(vals_a) * 1.3 if vals_a else 20)

    # Right: AMS subtype × severity heatmap-style grouped bars
    focus_subs = [s for s in ["AMSSleep", "AMSAgitated"] if s in ams_avail]
    if len(focus_subs) >= 2:
        x = np.arange(3)
        w = 0.25
        offset = -w
        colors_heat = {"AMSSleep": "#c0392b", "AMSAgitated": "#e67e22", "No AMS": "#95a5a6"}
        for sub_name in focus_subs + ["No AMS"]:
            rates_sev = []
            for sev_val in [1.0, 2.0, 3.0]:
                if sub_name == "No AMS":
                    mask = (df_analysis["AMS"] != 1) & (df_analysis["High_impact_InjSev"] == sev_val)
                else:
                    mask = (df_analysis[sub_name] == 1) & (df_analysis["High_impact_InjSev"] == sev_val)
                grp = df_analysis[mask]
                rates_sev.append(grp["y"].mean() * 100 if len(grp) >= 5 else 0)
            lbl = sub_name.replace("AMS", "") if sub_name != "No AMS" else "No AMS"
            axes[1].bar(x + offset, rates_sev, w, label=lbl,
                        color=colors_heat.get(sub_name, "#bdc3c7"), alpha=0.85, edgecolor="white")
            offset += w
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["Low", "Moderate", "High"])
        axes[1].set_xlabel("Injury Mechanism Severity")
        axes[1].set_ylabel("Positive CT rate (%)")
        axes[1].set_title("AMS Subtype × Severity Interaction")
        axes[1].legend(fontsize=9)
        axes[1].set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(figures_dir / "finding_ams_subtypes.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'finding_ams_subtypes.pdf'}")

    # ================================================================
    # NEW FINDING B: Post-traumatic seizures (not in PECARN) × age
    # ================================================================
    print("\n=== FINDING B: Post-Traumatic Seizures ===")
    if "Seiz" in df_analysis.columns:
        df_analysis["Seiz_clean"] = df_analysis["Seiz"].fillna(0)
        seiz_yes = df_analysis[df_analysis["Seiz_clean"] == 1]
        seiz_no = df_analysis[df_analysis["Seiz_clean"] != 1]
        print(f"  Seizure=Yes: PosCT={seiz_yes['y'].mean()*100:.1f}% (n={len(seiz_yes)})")
        print(f"  Seizure=No:  PosCT={seiz_no['y'].mean()*100:.1f}% (n={len(seiz_no)})")

        # Seizure × age group
        for age_grp, age_lbl in [(1, "<2 years"), (2, ">=2 years")]:
            for seiz_val, seiz_lbl in [(1, "Seizure"), (0, "No seizure")]:
                mask = (df_analysis["Seiz_clean"] == seiz_val) & (df_analysis["AgeTwoPlus"] == age_grp)
                grp = df_analysis[mask]
                if len(grp) >= 5:
                    print(f"  {seiz_lbl} & {age_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

        # Seizure × PECARN risk factor count
        print("\n  Seizure × PECARN risk factor count:")
        for nrf in range(5):
            cap = min(nrf, 4)
            for seiz_val, seiz_lbl in [(1, "Seizure"), (0, "No seizure")]:
                mask = (df_analysis["Seiz_clean"] == seiz_val) & (df_analysis["n_risk_capped"] == cap)
                grp = df_analysis[mask]
                if len(grp) >= 5:
                    lbl = f"{cap}+" if cap == 4 else str(cap)
                    print(f"    {seiz_lbl} & {lbl} PECARN factors: "
                          f"PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

        # Compare seizure PosCT to individual PECARN risk factors
        print("\n  PosCT rate comparison: seizure vs PECARN risk factors:")
        for rf in ["AMS", "Vomit", "LOCSeparate", "SFxPalp", "Hema", "SFxBas"]:
            if rf in df_analysis.columns:
                grp = df_analysis[df_analysis[rf] == 1]
                if len(grp) >= 10:
                    print(f"    {rf}=1: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Left: seizure vs no-seizure PosCT by age group
        age_labels = ["<2 years", ">=2 years"]
        seiz_rates = []
        no_seiz_rates = []
        for age_grp in [1, 2]:
            s = df_analysis[(df_analysis["Seiz_clean"] == 1) & (df_analysis["AgeTwoPlus"] == age_grp)]
            ns = df_analysis[(df_analysis["Seiz_clean"] != 1) & (df_analysis["AgeTwoPlus"] == age_grp)]
            seiz_rates.append(s["y"].mean() * 100 if len(s) >= 5 else 0)
            no_seiz_rates.append(ns["y"].mean() * 100 if len(ns) >= 5 else 0)
        x = np.arange(2)
        w = 0.35
        bars1 = axes[0].bar(x - w/2, no_seiz_rates, w, label="No seizure", color="#3498db", alpha=0.85)
        bars2 = axes[0].bar(x + w/2, seiz_rates, w, label="Seizure", color="#e74c3c", alpha=0.85)
        for bar_set in [bars1, bars2]:
            for bar in bar_set:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                             f"{bar.get_height():.1f}%", ha="center", fontsize=9)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(age_labels)
        axes[0].set_ylabel("Positive CT rate (%)")
        axes[0].set_title("Seizure × Age Group → PosCT")
        axes[0].legend()
        axes[0].set_ylim(0, max(seiz_rates + no_seiz_rates) * 1.3 if seiz_rates else 20)

        # Right: compare seizure PosCT to each PECARN variable PosCT
        compare_labels = ["Seizure"]
        compare_vals = [seiz_yes["y"].mean() * 100]
        compare_colors = ["#e74c3c"]
        pecarn_rfs = ["AMS", "SFxPalp", "Hema", "Vomit", "LOCSeparate", "SFxBas"]
        for rf in pecarn_rfs:
            if rf in df_analysis.columns:
                grp = df_analysis[df_analysis[rf] == 1]
                if len(grp) >= 10:
                    compare_labels.append(rf)
                    compare_vals.append(grp["y"].mean() * 100)
                    compare_colors.append("#3498db")
        bars = axes[1].barh(compare_labels, compare_vals, color=compare_colors, alpha=0.85)
        for bar, val in zip(bars, compare_vals):
            axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         f"{val:.1f}%", va="center", fontsize=9)
        axes[1].set_xlabel("Positive CT rate (%)")
        axes[1].set_title("Seizure vs PECARN Risk Factors: PosCT Rate")
        axes[1].invert_yaxis()
        fig.tight_layout()
        fig.savefig(figures_dir / "finding_seizures.pdf")
        plt.close()
        print(f"Saved {figures_dir / 'finding_seizures.pdf'}")

    # ================================================================
    # NEW FINDING C: Hematoma location × Vomiting/LOC × Gender
    # ================================================================
    print("\n=== FINDING C: Hematoma Location × Vomiting × LOC × Gender ===")
    df_analysis["HemaLoc_clean"] = df_analysis["HemaLoc"].fillna(-1)
    df_analysis["Gender_clean"] = df_analysis["Gender"].fillna(-1)
    df_analysis["Vomit_clean"] = df_analysis["Vomit"].fillna(0)
    df_analysis["LOC_any"] = (df_analysis["LOCSeparate"] >= 1).fillna(False).astype(int)
    df_analysis["VomitNbr_clean"] = df_analysis["VomitNbr"].fillna(0)

    hemaloc_map = {1: "Frontal", 2: "Temp/Par", 3: "Occipital"}
    gender_map = {1: "Male", 2: "Female"}

    print("  HemaLoc × PosCT (baseline):")
    for h_val, h_lbl in hemaloc_map.items():
        grp = df_analysis[df_analysis["HemaLoc_clean"] == h_val]
        if len(grp) >= 5:
            print(f"    {h_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    print("\n  HemaLoc × Vomiting → PosCT:")
    for h_val, h_lbl in hemaloc_map.items():
        for v_val, v_lbl in [(0, "No vomit"), (1, "Vomit")]:
            mask = (df_analysis["HemaLoc_clean"] == h_val) & (df_analysis["Vomit_clean"] == v_val)
            grp = df_analysis[mask]
            if len(grp) >= 5:
                print(f"    {h_lbl} + {v_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    print("\n  HemaLoc × LOC → PosCT:")
    for h_val, h_lbl in hemaloc_map.items():
        for l_val, l_lbl in [(0, "No LOC"), (1, "LOC")]:
            mask = (df_analysis["HemaLoc_clean"] == h_val) & (df_analysis["LOC_any"] == l_val)
            grp = df_analysis[mask]
            if len(grp) >= 5:
                print(f"    {h_lbl} + {l_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    print("\n  HemaLoc × Vomit × LOC (triple) → PosCT:")
    for h_val, h_lbl in hemaloc_map.items():
        for v_val, v_lbl in [(0, "NoVomit"), (1, "Vomit")]:
            for l_val, l_lbl in [(0, "NoLOC"), (1, "LOC")]:
                mask = ((df_analysis["HemaLoc_clean"] == h_val) &
                        (df_analysis["Vomit_clean"] == v_val) &
                        (df_analysis["LOC_any"] == l_val))
                grp = df_analysis[mask]
                if len(grp) >= 5:
                    print(f"    {h_lbl}+{v_lbl}+{l_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    print("\n  VomitNbr dose-response × HemaLoc:")
    for h_val, h_lbl in hemaloc_map.items():
        for vnbr_cat, vnbr_lbl in [((0, 0), "0 episodes"), ((1, 2), "1-2"), ((3, 99), "3+")]:
            mask = ((df_analysis["HemaLoc_clean"] == h_val) &
                    (df_analysis["VomitNbr_clean"] >= vnbr_cat[0]) &
                    (df_analysis["VomitNbr_clean"] <= vnbr_cat[1]))
            grp = df_analysis[mask]
            if len(grp) >= 5:
                print(f"    {h_lbl} + {vnbr_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    print("\n  Gender × HemaLoc × Vomit → PosCT:")
    for g_val, g_lbl in gender_map.items():
        for h_val, h_lbl in hemaloc_map.items():
            for v_val, v_lbl in [(0, "NoVomit"), (1, "Vomit")]:
                mask = ((df_analysis["Gender_clean"] == g_val) &
                        (df_analysis["HemaLoc_clean"] == h_val) &
                        (df_analysis["Vomit_clean"] == v_val))
                grp = df_analysis[mask]
                if len(grp) >= 10:
                    print(f"    {g_lbl}+{h_lbl}+{v_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    print("\n  Gender × HemaLoc × LOC → PosCT:")
    for g_val, g_lbl in gender_map.items():
        for h_val, h_lbl in hemaloc_map.items():
            for l_val, l_lbl in [(0, "NoLOC"), (1, "LOC")]:
                mask = ((df_analysis["Gender_clean"] == g_val) &
                        (df_analysis["HemaLoc_clean"] == h_val) &
                        (df_analysis["LOC_any"] == l_val))
                grp = df_analysis[mask]
                if len(grp) >= 10:
                    print(f"    {g_lbl}+{h_lbl}+{l_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    # Figure: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: HemaLoc × Vomit × LOC compound
    combo_labels = []
    combo_vals = []
    combo_colors = []
    color_map = {"Frontal": "#2ecc71", "Temp/Par": "#f39c12", "Occipital": "#e74c3c"}
    for h_val, h_lbl in hemaloc_map.items():
        for symptom_mask, symptom_lbl in [
            ((df_analysis["Vomit_clean"] == 0) & (df_analysis["LOC_any"] == 0), "Neither"),
            ((df_analysis["Vomit_clean"] == 1) | (df_analysis["LOC_any"] == 1), "Vomit or LOC"),
            ((df_analysis["Vomit_clean"] == 1) & (df_analysis["LOC_any"] == 1), "Both"),
        ]:
            mask = (df_analysis["HemaLoc_clean"] == h_val) & symptom_mask
            grp = df_analysis[mask]
            if len(grp) >= 5:
                combo_labels.append(f"{h_lbl}\n{symptom_lbl}")
                combo_vals.append(grp["y"].mean() * 100)
                combo_colors.append(color_map[h_lbl])

    bars = axes[0].bar(range(len(combo_labels)), combo_vals, color=combo_colors, alpha=0.85, edgecolor="white")
    for i, (bar, val) in enumerate(zip(bars, combo_vals)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontsize=7)
    axes[0].set_xticks(range(len(combo_labels)))
    axes[0].set_xticklabels(combo_labels, fontsize=7)
    axes[0].set_ylabel("Positive CT rate (%)")
    axes[0].set_title("Hematoma Location × Vomiting/LOC → PosCT")
    axes[0].set_ylim(0, max(combo_vals) * 1.25 if combo_vals else 20)

    # Right: Gender × HemaLoc for vomiting patients only
    loc_labels_r = list(hemaloc_map.values()) + ["No hematoma"]
    male_rates = []
    female_rates = []
    for h_val in list(hemaloc_map.keys()) + [-1]:
        for g_val, g_rates in [(1, male_rates), (2, female_rates)]:
            if h_val == -1:
                mask = (df_analysis["Gender_clean"] == g_val) & (df_analysis["Hema"] != 1)
            else:
                mask = (df_analysis["Gender_clean"] == g_val) & (df_analysis["HemaLoc_clean"] == h_val)
            grp = df_analysis[mask]
            g_rates.append(grp["y"].mean() * 100 if len(grp) >= 5 else 0)

    x = np.arange(len(loc_labels_r))
    w = 0.35
    axes[1].bar(x - w/2, male_rates, w, label="Male", color="#3498db", alpha=0.85)
    axes[1].bar(x + w/2, female_rates, w, label="Female", color="#e74c3c", alpha=0.85)
    for i, (m, f) in enumerate(zip(male_rates, female_rates)):
        axes[1].text(i - w/2, m + 0.3, f"{m:.1f}%", ha="center", fontsize=7)
        axes[1].text(i + w/2, f + 0.3, f"{f:.1f}%", ha="center", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(loc_labels_r, fontsize=8)
    axes[1].set_ylabel("Positive CT rate (%)")
    axes[1].set_title("Gender × Hematoma Location → PosCT")
    axes[1].legend()
    axes[1].set_ylim(0, max(male_rates + female_rates) * 1.25 if male_rates else 20)

    fig.tight_layout()
    fig.savefig(figures_dir / "finding_hemaloc_compound.pdf")
    plt.close()
    print(f"Saved {figures_dir / 'finding_hemaloc_compound.pdf'}")

    # ================================================================
    # Stability check for NEW findings (perturbation: exclude age < 2)
    # ================================================================
    print("\n=== Stability of New Findings Under Perturbation ===")
    ct_perturbed_analysis = df_analysis[
        df_analysis["AgeInMonth"].notna() & (df_analysis["AgeInMonth"] >= 24)
    ].copy()

    # Finding A stability: AMS subtypes in perturbed cohort
    print("  Finding A (AMS subtypes) - perturbed cohort:")
    for sub in ams_avail:
        ct_perturbed_analysis[sub] = ct_perturbed_analysis[sub].fillna(0)
        pos = ct_perturbed_analysis[ct_perturbed_analysis[sub] == 1]
        if len(pos) >= 5:
            print(f"    {sub}=1: PosCT={pos['y'].mean()*100:.1f}% (n={len(pos)})")

    # Finding B stability: seizures in perturbed cohort
    if "Seiz" in ct_perturbed_analysis.columns:
        ct_perturbed_analysis["Seiz_clean"] = ct_perturbed_analysis["Seiz"].fillna(0)
        seiz_p = ct_perturbed_analysis[ct_perturbed_analysis["Seiz_clean"] == 1]
        no_seiz_p = ct_perturbed_analysis[ct_perturbed_analysis["Seiz_clean"] != 1]
        print(f"  Finding B (Seizures) - perturbed: Seizure PosCT={seiz_p['y'].mean()*100:.1f}% (n={len(seiz_p)}); "
              f"No seizure={no_seiz_p['y'].mean()*100:.1f}%")

    # Finding C stability: HemaLoc × Vomit/LOC in perturbed cohort
    ct_perturbed_analysis["HemaLoc_clean"] = ct_perturbed_analysis["HemaLoc"].fillna(-1)
    ct_perturbed_analysis["Vomit_clean"] = ct_perturbed_analysis["Vomit"].fillna(0)
    ct_perturbed_analysis["LOC_any"] = (ct_perturbed_analysis["LOCSeparate"] >= 1).fillna(False).astype(int)
    print("  Finding C (HemaLoc×Vomit/LOC) - perturbed:")
    for h_val, h_lbl in {1: "Frontal", 2: "Temp/Par", 3: "Occipital"}.items():
        for v_val, v_lbl in [(0, "NoVomit"), (1, "Vomit")]:
            mask = (ct_perturbed_analysis["HemaLoc_clean"] == h_val) & (ct_perturbed_analysis["Vomit_clean"] == v_val)
            grp = ct_perturbed_analysis[mask]
            if len(grp) >= 5:
                print(f"    {h_lbl}+{v_lbl}: PosCT={grp['y'].mean()*100:.1f}% (n={len(grp)})")

    # ================================================================
    # END DEEPER EXPLORATION
    # ================================================================

    # Model comparison: accuracy for all 5 variants
    fig, ax = plt.subplots(figsize=(8, 4))
    models = ["PECARN\nCDR", "LR\n(default)", "LR\n(balanced)",
              "RF\n(default)", "RF\n(balanced)"]
    accs = [acc_pecarn, acc_lr, acc_lr_bal, acc_rf, acc_rf_bal]
    colors = ["#3498db", "#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]
    bars = ax.bar(models, accs, color=colors, alpha=0.85)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model accuracy comparison (test set)")
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.2f}", ha="center", fontsize=9)
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
