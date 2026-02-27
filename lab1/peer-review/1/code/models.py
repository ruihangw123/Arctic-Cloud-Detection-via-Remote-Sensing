from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

@dataclass(frozen=True)
class ModelResult:
    age_group: str
    model_name: str
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    sensitivity: float
    specificity: float
    npv: float     
    ct_rate: float
    extra: dict[str, Any]

def _make_Xy(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = d["ci_tbi"].astype(int).to_numpy()
    X = d.drop(columns=["ci_tbi", "age_group"], errors="ignore")
    X = X.select_dtypes(include=["number", "bool", "category"]).astype(float)
    return X, y

def _make_Xy_dt(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # y must be 0/1
    y = pd.to_numeric(d["ci_tbi"], errors="raise").astype(int).to_numpy()

    # drop obvious non-features and common metadata ids
    drop_cols = [
        "ci_tbi",
        "age_group",
        "pat_num",        
        "outcome_source",   
        "ci_tbi_source",    
    ]
    X = d.drop(columns=drop_cols, errors="ignore").copy()

    keep = X.select_dtypes(include=["number", "bool", "boolean"]).copy()

    cat_cols = X.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        s = pd.to_numeric(X[col].astype(str), errors="coerce")
        # keep only if almost everything converts (avoids string metadata cols)
        if s.notna().mean() >= 0.98:
            keep[col] = s

    # final coercion to float for sklearn
    keep = keep.apply(pd.to_numeric, errors="coerce").astype(float)

    return keep, y

def _summarize(y_true: np.ndarray, y_hat: np.ndarray) -> tuple[int, int, int, int, float, float, float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    
    # NPV calculation: True Negatives / (True Negatives + False Negatives)
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    
    ct_rate = float(np.mean(y_hat))
    
    return int(tn), int(fp), int(fn), int(tp), float(sens), float(spec), float(npv), float(ct_rate)

def pick_threshold_to_hit_sens(y_true: np.ndarray, proba: np.ndarray, target_sens: float) -> float:
    thresholds = np.unique(proba)
    thresholds.sort()
    if thresholds.size == 0:
        return 0.5
    best = float(thresholds[0])
    for thr in thresholds:
        y_hat = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens >= target_sens:
            best = float(thr)
    return best

def pick_threshold_max_spec(y_true: np.ndarray, proba: np.ndarray, target_sens: float) -> float:
    """
    Pick the threshold that maximizes specificity subject to sensitivity >= target_sens.
    Equivalent: pick the highest threshold that still achieves target_sens (ties handled).
    Returns ONLY the threshold (float).
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    if proba.size == 0:
        return 0.5

    thr_grid = np.unique(proba)
    thr_grid = np.r_[thr_grid, thr_grid.max() + 1e-12]  # allow predict-none

    best_thr = float("-inf")
    best_spec = float("-inf")

    for thr in thr_grid:
        y_hat = (proba >= thr).astype(int)

        tp = int(((y_true == 1) & (y_hat == 1)).sum())
        fn = int(((y_true == 1) & (y_hat == 0)).sum())
        tn = int(((y_true == 0) & (y_hat == 0)).sum())
        fp = int(((y_true == 0) & (y_hat == 1)).sum())

        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        if sens + 1e-12 >= target_sens:
            # maximize specificity; tie-breaker: higher threshold
            if (spec > best_spec + 1e-12) or (abs(spec - best_spec) <= 1e-12 and thr > best_thr):
                best_spec = spec
                best_thr = float(thr)

    if best_thr == float("-inf"):
        # cannot hit target sensitivity at any threshold
        # conservative fallback: predict all positive
        return float("-inf")

    return best_thr

def _map01(series, *, true_val=1, false_val=0):
    # preserves pd.NA if present (when dtype supports it)
    return series.map({true_val: True, false_val: False}).astype("boolean")

def create_pecarn_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Non-frontal scalp hematoma
    out["non_frontal_scalp_hematoma"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    hema = _map01(out["hema"])
    out.loc[hema.eq(False), "non_frontal_scalp_hematoma"] = False
    # hema_loc: 1 frontal, 2 occipital, 3 parietal/temporal 
    out.loc[hema.eq(True), "non_frontal_scalp_hematoma"] = out.loc[hema.eq(True), "hema_loc"].isin([2, 3]).astype("boolean")
    out.loc[hema.eq(True) & out["hema_loc"].isna(), "non_frontal_scalp_hematoma"] = pd.NA  # unknown location

    # LOC >= 5 sec for <2 (loc_len: 1 <5s, 2/3/4 >=5s) 
    out["loc_ge_5s"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    loc_sep = out["loc_separate"]
    out.loc[loc_sep.eq(0), "loc_ge_5s"] = False
    m = loc_sep.isin([1, 2])  # yes or suspected
    out.loc[m, "loc_ge_5s"] = out.loc[m, "loc_len"].isin([2, 3, 4]).astype("boolean")
    out.loc[m & out["loc_len"].isna(), "loc_ge_5s"] = pd.NA  # unknown duration

    # Severe headache: use severity when present; otherwise keep NA
    out["severe_headache"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    m = out["headache_severity"].notna()
    out.loc[m, "severe_headache"] = out.loc[m, "headache_severity"].eq(3).astype("boolean")

    # Severe mechanism: injury_severity: 1 low, 2 moderate, 3 high 
    out["severe_mechanism"] = out["injury_severity"].map({3: True, 1: False, 2: False}).astype("boolean")

    # Palpable skull fracture: 0 no, 1 yes, 2 unclear exam 
    out["palpable_skull_fx"] = out["skull_fx_palp"].map({1: True, 0: False, 2: True}).astype("boolean")

    out["not_acting_normally"] = out["act_norm"].map({0: True, 1: False}).astype("boolean")
    out["hist_of_vomiting"] = _map01(out["vomit"])
    out["signs_of_basilar_skull_fracture"] = _map01(out["skull_fx_bas"])

    # LOC for >=2 rule
    out["loc"] = out["loc_separate"].map({0: False, 1: True, 2: True}).astype("boolean")

    # Altered mental status: AMS is already a composite in the codebook
    out["altered_mental_status"] = out["ams"].map({1: True, 0: False}).astype("boolean")
    # If you want to force GCS=14 to count as altered even if AMS missing:
    out.loc[out["gcs_total"].eq(14), "altered_mental_status"] = True

    return out

def pecarn_under2_ct(d: pd.DataFrame, missing_preserve: bool=False) -> pd.Series:
    # recommend CT if any risk factor is present
    risk_flags = pd.concat(
        [
            d["altered_mental_status"],
            d["non_frontal_scalp_hematoma"],
            d["loc_ge_5s"],
            d["severe_mechanism"],
            d["palpable_skull_fx"],
            d["not_acting_normally"],
        ],
        axis=1,
    )
    if missing_preserve: # if True, then preserve NA as unknown; otherwise treat NA as positive
        any_true = risk_flags.eq(True).any(axis=1)
        any_missing = risk_flags.isna().any(axis=1)

        ct = pd.Series(False, index=risk_flags.index, dtype="boolean")
        ct[any_true] = True
        ct[(~any_true) & any_missing] = pd.NA
    else:
        # Treat missing clinical symptoms as absent (False)
        ct = risk_flags.fillna(False).any(axis=1).astype("boolean")
    return ct

def pecarn_two_plus_ct(d: pd.DataFrame, missing_preserve: bool=False) -> pd.Series:
    risk_flags = pd.concat(
        [
            d["altered_mental_status"],
            d["loc"],
            d["hist_of_vomiting"],
            d["severe_mechanism"],
            d["signs_of_basilar_skull_fracture"],
            d["severe_headache"],
        ],
        axis=1,
    )
    if missing_preserve:
        any_true = risk_flags.eq(True).any(axis=1)
        any_missing = risk_flags.isna().any(axis=1)

        ct = pd.Series(False, index=risk_flags.index, dtype="boolean")
        ct[any_true] = True
        ct[(~any_true) & any_missing] = pd.NA
    else:
        # Treat missing clinical symptoms as absent (False)
        ct = risk_flags.fillna(False).any(axis=1).astype("boolean")
    return ct    


# 1) Kuppermann CDR

def run_kuppermann_cdr(clean_df: pd.DataFrame, *, missing_preserve: bool=False, verbose: bool = False) -> list[ModelResult]:
    df = clean_df.copy()

    # PECARN restriction
    df = df.loc[df["gcs_total"].isin([14, 15])].copy()

    results: list[ModelResult] = []
    for grp in ["under_2", "two_plus"]:
        d = df[df["age_group"] == grp].copy()
        d = create_pecarn_predictors(d)
        cols_u2 = [
            "altered_mental_status",
            "non_frontal_scalp_hematoma",
            "loc_ge_5s",
            "severe_mechanism",
            "palpable_skull_fx",
            "not_acting_normally",
        ]
        cols_2p = [
            "altered_mental_status",
            "loc",
            "hist_of_vomiting",
            "severe_mechanism",
            "signs_of_basilar_skull_fracture",
            "severe_headache",
        ]

        def unknown_mask(df, cols):
            rf = df[cols]
            return (~rf.eq(True).any(axis=1)) & (rf.isna().any(axis=1))

        current_cols = cols_u2 if grp == "under_2" else cols_2p
        m = unknown_mask(d, current_cols)
        if verbose:
            print("unknown_rate:", m.mean())
            print(d.loc[m, current_cols].isna().mean().sort_values(ascending=False))


        y = d["ci_tbi"].astype(int).to_numpy()

        if grp == "under_2":
            ct = pecarn_under2_ct(d, missing_preserve=missing_preserve)          # pandas boolean with possible NA
            name = "kuppermann_cdr_under2"
        else:
            ct = pecarn_two_plus_ct(d, missing_preserve=missing_preserve)
            name = "kuppermann_cdr_two_plus"

        mask = ct.notna().to_numpy()
        ct_rate_overall = float(ct.fillna(True).astype(int).mean())
        ct_rate_decided = float(ct[ct.notna()].astype(int).mean()) if ct.notna().any() else float("nan")

        if mask.sum() == 0:
            tn = fp = fn = tp = 0
            sens = spec = float("nan")
        else:
            y_eval = y[mask]
            yhat_eval = ct[mask].astype(int).to_numpy()
            tn, fp, fn, tp, sens, spec, npv, ct_rate = _summarize(y_eval, yhat_eval)

        results.append(
            ModelResult(
                age_group=grp,
                model_name=name,
                threshold=float("nan"),
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
                sensitivity=sens,
                specificity=spec,
                ct_rate=ct_rate_overall,
                npv=npv,
                extra={
                    "n_total": int(len(ct)),
                    "n_unknown": int((~mask).sum()),
                    "unknown_rate": float((~mask).mean()),
                    "ct_rate_decided": ct_rate_decided,
                    "ct_rate_overall_conservative": ct_rate_overall,
                },
            )
        )

    return results

# 2) Logistic regression

def run_logistic_regression(
    df: pd.DataFrame,
    *,
    age_group: str,
    target_sens: float,
    random_state: int = 90,
) -> ModelResult:
    d = df[df["age_group"] == age_group].copy()
    X, y = _make_Xy(d)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
        
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
            ("model", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
            )),
        ]
    )

    param_grid = {
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best_params = gs.best_params_

    # Rebuild an unfitted pipeline with chosen hyperparameters for OOF thresholding
    best_pipe = gs.best_estimator_
    oof_proba = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=cv, method="predict_proba"
    )[:, 1]

    thr = pick_threshold_to_hit_sens(y_tr, oof_proba, target_sens)

    best_pipe.fit(X_tr, y_tr)
    imputer = best_pipe.named_steps["impute"]
    feature_names = imputer.get_feature_names_out(X_tr.columns)

    model = best_pipe.named_steps["model"]
    coefs = pd.Series(model.coef_.ravel(), index=feature_names, name="coef")
    selected_predictors = None
    if best_params["model__penalty"] == "l1":
        selected = coefs[coefs.abs() > 1e-8].sort_values(key=np.abs, ascending=False)
        selected_predictors = selected.index.to_list()

    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    y_hat = (proba_te >= thr).astype(int)

    tn, fp, fn, tp, sens, spec, npv, ct_rate = _summarize(y_te, y_hat)

    return ModelResult(
        age_group=age_group,
        model_name="logistic_regression",
        threshold=thr,
        tn=tn, fp=fp, fn=fn, tp=tp,
        sensitivity=sens,
        specificity=spec,
        npv=npv,
        ct_rate=ct_rate,
        extra={
            "predictors_all": list(feature_names),
            "predictors_selected": selected_predictors,
            "penalty": best_params["model__penalty"],
            "C": best_params["model__C"],
        },
    )

# 3) Decision tree

def run_decision_tree(
    clean_df: pd.DataFrame,
    *,
    age_group: str,
    target_sens: float,
    random_state: int = 90,
) -> ModelResult:
    d = clean_df[clean_df["age_group"] == age_group].copy()
    X, y = _make_Xy_dt(d)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    pipe = Pipeline(
        [
            # Maintain the clinical "charting by exception" assumption
            ("impute", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
            ("tree", DecisionTreeClassifier(random_state=random_state)),
        ]
    )

    param_grid = {
        "tree__max_depth": [2, 3, 4, 5],
        "tree__min_samples_leaf": [50, 100, 200],
        "tree__min_samples_split": [200, 500, 1000],
        "tree__ccp_alpha": [0.0, 1e-3, 1e-2, 1e-1],
        "tree__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best_pipe: BaseEstimator = gs.best_estimator_

    # OOF proba on TRAIN ONLY for threshold selection
    oof_proba = cross_val_predict(best_pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = pick_threshold_max_spec(y_tr, oof_proba, target_sens)

    best_pipe.fit(X_tr, y_tr)

    imputer = best_pipe.named_steps["impute"]
    feature_names = imputer.get_feature_names_out(X_tr.columns)

    tree = best_pipe.named_steps["tree"]
    importances = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_importances = importances.head(25)

    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    y_hat = (proba_te >= thr).astype(int)

    tn, fp, fn, tp, sens, spec, npv, ct_rate = _summarize(y_te, y_hat)

    return ModelResult(
        age_group=age_group,
        model_name="decision_tree",
        threshold=float(thr),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        sensitivity=sens,
        specificity=spec,
        npv=npv,
        ct_rate=ct_rate,
        extra={
            "best_params": gs.best_params_,
            "top_importances": top_importances.to_dict(),
        },
    )

def run_all_models(df: pd.DataFrame, target_sens: float = 0.95, random_state: int = 90) -> list[ModelResult]:
    results = []
    for age_group in ["under_2", "two_plus"]:
        res_lr = run_logistic_regression(df, age_group=age_group, target_sens=target_sens, random_state=random_state)
        res_dt = run_decision_tree(df, age_group=age_group, target_sens=target_sens, random_state=random_state)
        results.extend([res_lr, res_dt])

    results.extend(run_kuppermann_cdr(df, missing_preserve=False, verbose=False))
    return results



