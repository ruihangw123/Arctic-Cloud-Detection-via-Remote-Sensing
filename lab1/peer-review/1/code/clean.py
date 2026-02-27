"""
Data cleaning and feature engineering for the PECARN TBI dataset.

Design goals
- Standardize column names to a consistent format.
- Convert documented NA codes (92 = not applicable) to pd.NA.
  Only apply to columns where the codebook defines 92 as not applicable.
- Enforce logical consistency between parent and child fields.
- Add indicators related to CT completion and findings.
- Add a composite ciTBI outcome from pos_int_final or a reconstructed definition.
- Optionally drop post-outcome / post-CT columns to reduce leakage in modeling.
"""
import re
from typing import Iterable
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Constants and configuration
NA_CODE_DEFAULT: tuple[int, ...] = (92,)  # 92 = not applicable (structured NA)

FINDINGS_COLS: list[str] = [f"finding{i}" for i in range(1, 24)]

# These are columns that are *outcomes* or post-CT quantities and should not be used as predictors.
OUTCOME_COLS: tuple[str, ...] = (
    "pos_int_final",
    "neurosurgery",
    "intub24_head",
    "death_tbi",
    "hosp_head",
    "hosp_head_pos_ct",
    "ct_form1",
)

# Columns that should remain numeric in the final cleaned output.
TRUE_NUMERIC_STD_VARS: set[str] = {"pat_num", "age_months", "age_years", "gcs_total"}

# Standardized versions produced by standardize_col_names().
TRUE_BOOLEAN_STD_VARS: set[str] = {
    "pos_int_final",
    "neurosurgery",
    "intub24_head",
    "hosp_head_pos_ct",
    "hosp_head",
    "death_tbi",
    "ct_done",
    "observed",
    "ct_form1",
    "drugs",
    "osi",
    "neuro_def",
    "clav",
    "hema",
    "skull_fx_bas",
    "font_bulg",
    "ams",
    "dizzy",
    "intubated",
    "paralyzed",
    "sedated",
    "vomit",
    "act_norm",
    "seiz",
}

# Special 3-level verb variables: 0=no, 1=yes, 91=preverbal/nonverbal.
VERB_TRILEVEL_STD_VARS: set[str] = {"amnesia_verb", "headache_verb"}
VERB_TRILEVEL_CATS: list[int] = [0, 1, 91]


NA92_RAW_VARS = {
    "AMSAgitated", "AMSOth", "AMSRepeat", "AMSSleep", "AMSSlow",
    "CTSed", "CTSedAge", "CTSedAgitate", "CTSedOth", "CTSedRqst",
    "ClavFace", "ClavFro", "ClavNeck", "ClavOcc", "ClavPar", "ClavTem",
    "EDCT",
    "Finding1", "Finding2", "Finding3", "Finding4", "Finding5", "Finding6", "Finding7",
    "Finding8", "Finding9", "Finding10", "Finding11", "Finding12", "Finding13", "Finding14",
    "Finding20", "Finding21", "Finding22", "Finding23",
    "HASeverity", "HAStart",
    "HemaLoc", "HemaSize",
    "IndAMS", "IndAge", "IndAmnesia", "IndClinSFx", "IndHA", "IndHema", "IndLOC",
    "IndMech", "IndNeuroD", "IndOth", "IndRqstMD", "IndRqstParent", "IndRqstTrauma",
    "IndSeiz", "IndVomit", "IndXraySFx",
    "LocLen",
    "NeuroDCranial", "NeuroDMotor", "NeuroDOth", "NeuroDReflex", "NeuroDSensory",
    "OSIAbdomen", "OSICspine", "OSICut", "OSIExtremity", "OSIFlank", "OSIOth", "OSIPelvis",
    "PosCT",
    "SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi",
    "SFxPalpDepress",
    "SeizLen", "SeizOccur",
    "VomitLast", "VomitNbr", "VomitStart",
}

# parent - child columns
BASE_GATES = {
    "neuro_def": ["neuro_def_motor","neuro_def_sensory","neuro_def_cranial","neuro_def_reflex","neuro_def_oth"],
    "clav": ["clav_face","clav_neck","clav_fro","clav_occ","clav_par","clav_tem"],
    "hema": ["hema_loc","hema_size"],
    "ams": ["ams_agitated","ams_sleep","ams_slow","ams_repeat","ams_oth"],
    "seiz": ["seiz_occur","seiz_len"],
    "loc_separate": ["loc_len"],
    "headache_verb": ["headache_severity","headache_start"],
    "vomit": ["vomit_num","vomit_start","vomit_last"],
    "skull_fx_palp": ["skull_fx_palp_depress"],
    "skull_fx_bas": ["skull_fx_bas_hem","skull_fx_bas_oto","skull_fx_bas_per","skull_fx_bas_ret","skull_fx_bas_rhi"],
    "osi": ["osi_extremity","osi_cut","osi_cspine","osi_flank","osi_abdomen","osi_pelvis","osi_oth"],
}

# for each parent, the codes meaning "yes", "no", and optionally "not assessable" (e.g. preverbal/nonverbal or not applicable)
BASE_RULES = {
    "neuro_def": {"yes": [1], "no": [0]},
    "clav": {"yes": [1], "no": [0]},
    "hema": {"yes": [1], "no": [0]},
    "ams": {"yes": [1], "no": [0]},
    "seiz": {"yes": [1], "no": [0]},
    "loc_separate": {"yes": [1,2], "no": [0]},
    "headache_verb": {"yes": [1], "no": [0], "not_assessable": [91]},
    "vomit": {"yes": [1], "no": [0]},
    "skull_fx_palp": {"yes": [1], "no": [0,2]}, # 2 is unclear, but child is marked as 92 if unclear, no or missing
    "skull_fx_bas": {"yes": [1], "no": [0]}, 
    "osi": {"yes": [1], "no": [0]},
}

# Parents where yes should imply at least one detail is positive (otherwise treat as inconsistent).
REQUIRE_ONE_YES = {"neuro_def", "clav", "osi", "skull_fx_bas"}

def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    explicit = {
        "empl_type": "employ_type",
        "high_impact__inj_sev": "injury_severity",
        "injury_mech": "injury_mechanism",
        "o_s_i": "osi",
        "g_c_s": "gcs",
        "a_m_s": "ams",
        "h_a_verb": "headache_verb",
        "l_o_c_separate": "loc_separate",
        "vomit_nbr": "vomit_num",
        "agein_years": "age_years",
        "age_in_month": "age_months",
        "neuro_d": "neuro_def",
    }

    rules = [
        ("g_c_s_", "gcs_"),
        ("a_m_s_", "ams_"),
        ("h_a_", "headache_"),
        ("l_o_c_", "loc_"),
        ("c_t_", "ct_"),
        ("e_d_", "ed_"),
        ("ind_a_m_s", "ind_ams"),
        ("ind_l_o_c", "ind_loc"),
        ("ind_h_a", "ind_headache"),
        ("ind_rqst_m_d", "ind_rqst_md"),
        ("ind_xray_s_fx", "ind_xray_skull_fx"),
        ("ind_clin_s_fx", "ind_clin_skull_fx"),
        ("death_t_b_i", "death_tbi"),
        ("pos_c_t", "pos_ct"),
        ("ed_c_t", "ed_ct"),
        ("c_t_form1", "ct_form1"),
    ]

    def _normalize_one(col: object) -> str:
        name = str(col).strip()
        name = re.sub(r"[\s\.\-]+", "_", name)
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

        for old, new in rules:
            name = name.replace(old, new)

        if name.startswith("s_fx_"):
            name = "skull_fx_" + name[len("s_fx_") :]
        if name.startswith("neuro_d_"):
            name = "neuro_def_" + name[len("neuro_d_") :]
        if name.startswith("o_s_i_"):
            name = "osi_" + name[len("o_s_i_") :]

        return explicit.get(name, name)

    out = df.copy()
    out.columns = [_normalize_one(c) for c in df.columns]
    return out

def _standardize_name_map(raw_names: Iterable[str]) -> dict[str, str]:
    """Map raw codebook names to standardized names (via standardize_col_names)."""
    tmp = pd.DataFrame(columns=list(raw_names))
    tmp2 = standardize_col_names(tmp)
    return dict(zip(tmp.columns, tmp2.columns))


# Precompute standardized columns where 92 means "Not applicable".
_NA92_STD_COLS: set[str] = set(_standardize_name_map(NA92_RAW_VARS).values())


def apply_na92_only_where_documented(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 92 -> NA only where codebook says 92 is not applicable."""
    out = df.copy()
    for col in _NA92_STD_COLS:
        if col in out.columns:
            out[col] = out[col].replace(92, pd.NA)
    return out

# feature engineering
def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add age_group column based on age_two_plus.

    Output categories:
    - 1: "under_2" 
    - 2: "two_plus" 
    """
    out = df.copy()

    if "age_two_plus" not in out.columns:
        return out

    # 1 = <2 years, 2 = >=2 years
    out["age_group"] = out["age_two_plus"].map({1: "under_2", 2: "two_plus"})

    # if age_two_plus is present but has unexpected values, set age_group to NA for those rows
    bad = out["age_two_plus"].notna() & ~out["age_two_plus"].isin([1, 2])
    out.loc[bad, "age_group"] = pd.NA
    return out

def fix_gcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate GCS total from components when all components are present.
    Flag rows where components are present but total is inconsistent. 
    This can help catch data entry errors in GCS total or components.

    Adds: 
    - gcs_total_calc: recalculated total from components when all components are present.
    - bad_gcs_flag: True if components are present but total is inconsistent.
    """
    out = df.copy()
    gcs_dep = ["gcs_eye","gcs_verbal","gcs_motor"]

    # check all components are present
    if not all(col in out.columns for col in gcs_dep):
        return out

    # Create mask for rows where all GCS components are present
    mask = out[gcs_dep].notna().all(axis=1)
    # for all rows where components are present, calculate total
    out.loc[mask, "gcs_total_calc"] = out.loc[mask, gcs_dep].sum(axis=1)

    # flag rows where components are present but total is inconsistent
    if "gcs_total" in out.columns:
        out["bad_gcs_flag"] = mask & out["gcs_total"].notna() & (out["gcs_total"] != out["gcs_total_calc"])
    else:
        out["bad_gcs_flag"] = pd.NA
    return out

def apply_gcs_total_strategy(df: pd.DataFrame, *, strategy: str = "keep") -> pd.DataFrame:
    """
    Decide what to do when gcs_total disagrees with the sum of components.

    strategy:
    - "keep": keep original gcs_total; just flag inconsistencies
    - "replace": replace gcs_total with gcs_total_calc where inconsistent
    - "set_na": set gcs_total to NA where inconsistent
    """
    out = df.copy()
    if "bad_gcs_flag" not in out.columns or "gcs_total" not in out.columns or "gcs_total_calc" not in out.columns:
        return out

    bad = out["bad_gcs_flag"].fillna(False)
    if strategy == "keep":
        return out
    if strategy == "replace":
        out.loc[bad, "gcs_total"] = out.loc[bad, "gcs_total_calc"]
        return out
    if strategy == "set_na":
        out.loc[bad, "gcs_total"] = pd.NA
        return out

    raise ValueError("gcs_total_strategy must be one of: 'keep', 'replace', 'set_na'.")

# CT indicators and CT-dependent fields
def add_ct_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ct_done" in out.columns:
        out["ct_was_done"] = out["ct_done"].map({1: True, 0: False}).astype("boolean")
    else:
        out["ct_was_done"] = pd.NA

    # ED CT subtype (only meaningful when CT was done)
    if "ed_ct" in out.columns:
        out["ed_ct_in_ed"] = out["ed_ct"].map({1: True, 0: False}).astype("boolean")
        out.loc[out["ct_was_done"].eq(False), "ed_ct_in_ed"] = pd.NA

    return out

def get_ct_depend_cols(df: pd.DataFrame) -> list[str]:
    """
    Return all columns that should be meaningful when CT was done.
    Includes:
    - finding1-23
    - indication columns (starting with ind_)
    - ct sedation columns (starting with ct_sed_)
    - post_ct and ed_ct 
    """

    cols = [c for c in (FINDINGS_COLS + ["pos_ct", "ed_ct"]) if c in df.columns]
    # add indication columns
    cols.extend([c for c in df.columns if c.startswith("ind_")])
    # add sedation columns
    cols.extend([c for c in df.columns if c.startswith("ct_sed")])

    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def flag_ct_dependency_violations(df: pd.DataFrame, *, ct_depend_cols: list[str]) -> pd.DataFrame:
    """
    Flag rows where CT-dependent fields are present but CT was not done
    """
    out = df.copy()
    if "ct_was_done" not in out.columns:
        out["ct_dependency_violation"] = pd.NA
        return out
    
    ct_cols = [c for c in ct_depend_cols if c in out.columns]
    if not ct_cols:
        out["ct_dependency_violation"] = False
        return out
    
    no_ct = out["ct_was_done"].eq(False)
    ct_known = out["ct_was_done"].notna()
    has_ct_info = out[ct_cols].notna().any(axis=1)
    out["ct_dependency_violation"] = ct_known & no_ct & has_ct_info
    return out

def handle_ct_dependent_fields(df: pd.DataFrame, *, ct_depend_cols: list[str]) -> pd.DataFrame:
    """
    If CT was not done, set all CT-dependent fields to NA.
    """
    out = df.copy()
    if "ct_was_done" not in out.columns:
        return out

    mask_no_ct = out["ct_was_done"].eq(False)
    for col in ct_depend_cols:
        if col in out.columns:
            out.loc[mask_no_ct, col] = pd.NA
    return out

# Parent-child consistency
def build_gates_and_rules_for_df(
    df: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[int]]]]:
    """Extend base gates/rules with CT-form gates when present."""
    gates = {k: v[:] for k, v in BASE_GATES.items()}
    rules = {k: {kk: vv[:] for kk, vv in v.items()} for k, v in BASE_RULES.items()}

    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    ctsed_children = [c for c in df.columns if c.startswith("ct_sed_")]

    if "ct_form1" in df.columns:
        children = ind_cols[:]
        if "ct_sed" in df.columns:
            children.append("ct_sed")
        gates["ct_form1"] = children
        rules["ct_form1"] = {"yes": [1], "no": [0]}

    if "ct_sed" in df.columns:
        gates["ct_sed"] = ctsed_children
        rules["ct_sed"] = {
            "yes": [1],
            "no": [0],
            "not_assessable": list(NA_CODE_DEFAULT),
        }

    return gates, rules

def fix_parent_child(
    df: pd.DataFrame,
    *,
    gates: dict[str, list[str]],
    rules: dict[str, dict[str, list[int]]],
    require_one_yes: set[str],
    enforce_detail_presence: bool = False,
) -> pd.DataFrame:
    """Apply parent-child consistency rules described in the module docstring."""
    out = df.copy()

    for parent, children in gates.items():
        if parent not in out.columns:
            continue

        valid_children = [c for c in children if c in out.columns]
        if not valid_children:
            continue

        r = rules[parent]
        yes_codes = r.get("yes", [1])
        no_codes = set(r.get("no", []))
        na_codes = set(r.get("not_assessable", []))
        wipe_codes = no_codes | na_codes

        child_present = out[valid_children].notna().any(axis=1)
        infer_parent_yes = out[parent].isna() & child_present
        if infer_parent_yes.any():
            out.loc[infer_parent_yes, parent] = yes_codes[0]

        p = out[parent]
        wipe_children = p.isna() | p.isin(wipe_codes)
        out.loc[wipe_children, valid_children] = pd.NA

        if parent in require_one_yes:
            p_yes = out[parent].isin(yes_codes)
            children_all_zero = out[valid_children].eq(0).all(axis=1, skipna=False)
            mask = p_yes & children_all_zero
            if mask.any():
                out.loc[mask, parent] = 0
                out.loc[mask, valid_children] = pd.NA

        if enforce_detail_presence:
            p_yes = out[parent].isin(yes_codes)
            all_missing = out[valid_children].isna().all(axis=1)
            out.loc[p_yes & all_missing, parent] = pd.NA

    return out

# Outcome
def add_citbi_outcome(
    df: pd.DataFrame,
    *,
    outcome_source: str = "pos_int_final",
    audit_threshold: float | None = None, 
    keep_audit_cols: bool = False) -> pd.DataFrame:
    """
    Add a boolean ciTBI outcome column based on the official pos_int_final variable or a reconstructed version based on the component criteria.

    outcome_source options:
        - "pos_int_final": use the official variable directly (mapped to boolean)
        - "reconstructed": create ciTBI based on the component criteria 

    Adds: 
        - ci_tbi: boolean/NA
        - ci_tbi_source: string label indicating which source was used ("pos_int_final" or "reconstructed")
    Optional audit:
        - ci_tbi_rebuilt: the reconstructed ciTBI based on components (for auditing purposes)
        - ci_tbi_mismatch_flag: boolean flag indicating rows where pos_int_final and reconstructed ciTBI do not match 
    """
    out = df.copy()

    needed = ["neurosurgery", "intub24_head", "death_tbi", "hosp_head", "pos_ct"]
    can_rebuild = all(c in out.columns for c in needed)

    if can_rebuild:
        complete = out[needed].notna().all(axis=1)
        out["ci_tbi_rebuilt"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        out.loc[complete, "ci_tbi_rebuilt"] = (
        out.loc[complete, "neurosurgery"].eq(1)
        | out.loc[complete, "intub24_head"].eq(1)
        | out.loc[complete, "death_tbi"].eq(1)
        | (out.loc[complete, "hosp_head"].eq(1) & out.loc[complete, "pos_ct"].eq(1)))
    else:
        out["ci_tbi_rebuilt"] = pd.NA

    # choose outcome used for modeling
    if outcome_source == "pos_int_final":
        if "pos_int_final" not in out.columns:
            raise KeyError("pos_int_final not found, cannot use outcome_source='pos_int_final'.")
        out["ci_tbi"] = out["pos_int_final"].map({1: True, 0: False}).astype("boolean")
        out["ci_tbi_source"] = "pos_int_final"

    elif outcome_source == "reconstructed":
        if not can_rebuild:
            missing = [c for c in needed if c not in out.columns]
            raise KeyError(f"Cannot reconstruct ciTBI, missing columns: {missing}")
        out["ci_tbi"] = out["ci_tbi_rebuilt"].astype("boolean")
        out["ci_tbi_source"] = "reconstructed"

    else:
        raise ValueError("outcome_source must be 'pos_int_final' or 'reconstructed'.")

    # audit against official if both exist
    if audit_threshold is not None and "pos_int_final" in out.columns and can_rebuild:
        official = out["pos_int_final"].map({1: True, 0: False})
        mismatch = (out["ci_tbi_rebuilt"].notna()) & (out["ci_tbi_rebuilt"] != official)
        rate = float(mismatch.mean())
        out["ci_tbi_mismatch_flag"] = mismatch

        if rate > audit_threshold:
            raise ValueError(
                f"ciTBI mismatch rate {rate:.4%} exceeds threshold {audit_threshold:.4%}"
            )
    else:
        out["ci_tbi_mismatch_flag"] = pd.NA

    # drop audit columns if not needed
    if not keep_audit_cols:
        out = out.drop(columns=["ci_tbi_rebuilt", "ci_tbi_mismatch_flag"], errors="ignore")

    return out

# Modeling leakage reduction
def drop_cols_for_modeling(df: pd.DataFrame, *, target_col: str = "ci_tbi") -> pd.DataFrame:
    out = df.copy()

    leakage_exact = {
        "pat_num",
        "pos_int_final",
        "neurosurgery", "intub24_head", "death_tbi",
        "hosp_head", "hosp_head_pos_ct", "pos_ct",
        "ci_tbi_source", "ci_tbi_rebuilt", "ci_tbi_mismatch_flag",
        "gcs_total_calc", "bad_gcs_flag", "ct_dependency_violation",
        "ct_done", "ct_was_done", "ed_ct", "ct_form1",
        "ed_ct_in_ed", "ed_disposition", "drugs", "intubated", "paralyzed", "sedated",
        "observed", "employ_type", "certification",
        "race", "ethnicity", "gender", "age_months", "age_years", "age_two_plus",
    }

    leakage_exact.discard(target_col)

    leakage_prefixes = ("finding", "ind_", "ct_", "ed_")  # drop all CT-related and findings

    cols_to_drop = []
    for c in out.columns:
        if c in leakage_exact or c.startswith(leakage_prefixes):
            if c != target_col:
                cols_to_drop.append(c)

    return out.drop(columns=cols_to_drop, errors="ignore")

# Dtypes 
def _to_boolean01(s: pd.Series, *, colname: str) -> pd.Series:
    """Coerce a 0/1-coded flag to pandas BooleanDtype."""
    if s.dtype.name == "boolean":
        return s

    x = s.dropna()
    if not x.empty:
        allowed = {0, 1, 0.0, 1.0, False, True}
        bad = x.loc[~x.isin(allowed)]
        if not bad.empty:
            vc = bad.value_counts().head(10).to_dict()
            raise ValueError(
                f"Unexpected codes in boolean column '{colname}': {vc}. "
                "Expected only 0/1.",
            )

    mapped = s.map({0: False, 1: True})
    return mapped.astype("boolean")


def coerce_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final dtype rules requested for this project."""
    out = df.copy()

    keep_numeric = set(TRUE_NUMERIC_STD_VARS)
    keep_boolean = set(TRUE_BOOLEAN_STD_VARS) | {"ci_tbi"}

    for col in sorted(keep_boolean):
        if col in out.columns:
            out[col] = _to_boolean01(out[col], colname=col)

    for col in VERB_TRILEVEL_STD_VARS:
        if col not in out.columns:
            continue
        bad = out[col].dropna().loc[~out[col].isin(VERB_TRILEVEL_CATS)]
        if not bad.empty:
            vc = bad.value_counts().head(10).to_dict()
            raise ValueError(
                (
                    f"Unexpected codes in '{col}': {vc}. Expected only "
                    f"{list(VERB_TRILEVEL_CATS)}."
                )
            )
        out[col] = pd.Categorical(out[col], categories=list(VERB_TRILEVEL_CATS))

    # Convert remaining columns to category (except numeric + boolean + tri-level).
    for col in out.columns:
        if col in keep_numeric or col in keep_boolean or col in VERB_TRILEVEL_STD_VARS:
            continue
        out[col] = out[col].astype("category")

    # Ensure numeric columns stay numeric (avoid accidental category from upstream).
    for col in keep_numeric:
        if col in out.columns and not is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

# Main function 
def clean_data(df: pd.DataFrame, *,
    drop_missing_over: float | None = None,
    outcome_source: str = "pos_int_final",   # or reconstructed -> can verify how much results depend on using the official variable vs reconstructing from components
    modeling: bool = False, 
    enforce_detail_presence: bool = False, # if True, set parent to NA when parent is yes but all children are missing -> for perturbation analysis to see how much results depend on those cases (who may have data quality issues)
    audit_threshold: float | None = 0.001,   
    convert_dtypes: bool = False, # if True, apply final dtype coercion to numeric, boolean, and categorical types; if False, keep dtypes as is (which may be more permissive but less consistent)
    gcs_total_strategy: str = "replace", # options: "keep" (default), "replace", "set_na" -> for perturbation analysis to see how much results depend on how we handle GCS inconsistencies
    warn_ranges: bool = False,
    ) -> pd.DataFrame:
    """
    Main data cleaning function that applies all steps in sequence.

    Parameters:
    ----------
    - df: input dataframe
    - doc_xlsx_path: optional path to documentation Excel file for building missing code mapping; if None, use default_na_values
    - drop_missing_over: if not None, drop columns with more than this proportion of missing
    - outcome_source: which ciTBI outcome source to use ("pos_int_final" or "reconstructed")
    - modeling: if True, drop non-predictive outcome columns and audit columns for modeling
    - enforce_detail_presence: if True, set parent to NA when parent is yes but all children are missing (can help catch data quality issues)
    - audit_threshold: if not None, raise error if reconstructed ciTBI disagrees with pos_int_final more than this proportion of the time (requires both to be present)
    - convert_dtypes: if True, apply final dtype coercion to numeric, boolean, and categorical types; if False, keep dtypes as is (which may be more permissive but less consistent)
    - gcs_total_strategy: how to handle GCS total when it disagrees with sum of components ("keep", "replace", "set_na")
    - warn_ranges: if True, print warnings about out-of-range values for key variables; if False, just set them to NA without warning
    
    Returns:
    -------
    pd.DataFrame:cleaned dataframe ready for analysis or modeling with audit columns preserved if modeling is False.
    """
    out = standardize_col_names(df)
    out = apply_na92_only_where_documented(out)

    # Patient id uniqueness check
    if "pat_num" in out.columns and not out["pat_num"].is_unique:
        raise ValueError("pat_num is not unique")

    # optionally drop columns with too much missing data
    if drop_missing_over is not None:
        out = out.loc[:, out.isna().mean() < drop_missing_over]

    out = fix_gcs(out) # now data contains gcs_total_calc and bad_gcs_flag
    out = apply_gcs_total_strategy(out, strategy=gcs_total_strategy)
    out = add_age_group(out) 

    # CT-related indicators and consistency checks
    out = add_ct_indicators(out)
    ct_depend_cols = get_ct_depend_cols(out)
    out = flag_ct_dependency_violations(out, ct_depend_cols=ct_depend_cols)
    out = handle_ct_dependent_fields(out, ct_depend_cols=ct_depend_cols)

    # parent-child consistency 
    gates, rules = build_gates_and_rules_for_df(out)
    out = fix_parent_child(
        out,
        gates=gates,
        rules=rules,
        require_one_yes=REQUIRE_ONE_YES,
        enforce_detail_presence=enforce_detail_presence,
    )

    # outcome for later modeling
    out = add_citbi_outcome(
        out,
        outcome_source=outcome_source,
        audit_threshold=audit_threshold,
        keep_audit_cols=not modeling
    )
    out = out.loc[out["ci_tbi"].notna()].copy()
 
    # drop non-predictive outcome columns for modeling
    if modeling:
        out = drop_cols_for_modeling(out, target_col="ci_tbi")

    out = out.loc[out["ci_tbi"].notna()].copy()

    if convert_dtypes:
        out = coerce_output_dtypes(out)

    return out



