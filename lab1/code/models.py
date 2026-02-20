"""
Classification models for PECARN TBI CT scan recommendation.
Implements: Kuppermann clinical decision rule, logistic regression, and random forest.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
        log_loss,
    )

    acc = accuracy_score(y_true, y_pred)
    proba = model.predict_proba(scaler.transform(
        np.zeros((1, len(feature_names)))
    ))  # dummy, not stored; we recompute below
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
