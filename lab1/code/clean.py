"""
Data cleaning functions for PECARN TBI dataset.

Per TBI PUD Documentation 10-08-2013.xlsx:
- 91 = Pre-verbal/Non-verbal (patient unable to respond)
- 92 = Not applicable (e.g., LOC length when no LOC)
- 99 = Unknown/Refused
- 90 = Other (for categorical variables)
- -1 = Missing/Refused (sometimes used)
"""

import pandas as pd
import numpy as np


# Missing/special codes per documentation
MISSING_CODES = {91, 92, 99, 90, -1}


def clean_data(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Clean the raw PECARN TBI DataFrame.

    Steps:
    1. Replace documented missing/special codes (91, 92, 99, 90, -1) with NaN
    2. Handle empty strings as missing
    3. Preserve patient identifiers

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as loaded by pd.read_csv()
    inplace : bool, default False
        If True, modify df in place. Otherwise return a copy.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    if not inplace:
        df = df.copy()

    # Replace standard missing codes in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "PatNum":
            continue
        mask = df[col].isin(MISSING_CODES)
        if mask.any():
            df.loc[mask, col] = np.nan

    # Handle object/string columns - treat empty and common codes as missing
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace({"": np.nan, " ": np.nan})
        # Some object cols may have numeric codes stored as strings
        try:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            mask = numeric_vals.isin(MISSING_CODES)
            df.loc[mask, col] = np.nan
        except (TypeError, ValueError):
            pass

    return df


def flag_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add consistency-check flags. Does not modify original data.

    Flags:
    - GCS_mismatch: GCSEye + GCSVerbal + GCSMotor != GCSTotal
    - Age_mismatch: AgeTwoPlus conflicts with AgeInMonth-derived age group

    Returns
    -------
    pd.DataFrame
        Copy of df with added flag columns
    """
    df = df.copy()
    df["GCS_mismatch"] = False
    df["Age_mismatch"] = False

    # GCS: components should sum to total (when all present)
    if all(c in df.columns for c in ["GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal"]):
        gcs_sum = df["GCSEye"] + df["GCSVerbal"] + df["GCSMotor"]
        valid = df["GCSTotal"].notna() & df["GCSEye"].notna()
        df.loc[valid & (gcs_sum != df["GCSTotal"]), "GCS_mismatch"] = True

    # Age: AgeTwoPlus (1=<2y, 2=>=2y) vs AgeInMonth
    if "AgeInMonth" in df.columns and "AgeTwoPlus" in df.columns:
        valid_age = df["AgeInMonth"].notna() & (df["AgeInMonth"] >= 0) & (df["AgeInMonth"] <= 300)
        under2_from_month = (df.loc[valid_age, "AgeInMonth"] / 12 >= 2).astype(int) + 1
        expected = under2_from_month
        actual = df.loc[valid_age, "AgeTwoPlus"]
        df.loc[valid_age & (actual != expected), "Age_mismatch"] = True

    return df


def derive_ciTBI(df: pd.DataFrame) -> pd.Series:
    """
    Derive clinically-important TBI (ciTBI) per Kuppermann et al. definition.

    ciTBI = death from TBI OR neurosurgery OR intubation >24h for TBI
            OR positive intracranial injury on final read (PosIntFinal).

    Uses: DeathTBI, Neurosurgery, Intub24Head, PosIntFinal.
    """
    result = np.zeros(len(df), dtype=int)
    for col in ["DeathTBI", "Neurosurgery", "Intub24Head", "PosIntFinal"]:
        if col in df.columns:
            result = np.maximum(result, df[col].eq(1).fillna(False).astype(int).values)
    return pd.Series(result, index=df.index)


def get_analysis_cohort(
    df: pd.DataFrame,
    gcs_range: tuple[int, int] = (14, 15),
    require_ct_or_outcome: bool = True,
) -> pd.DataFrame:
    """
    Restrict to analysis cohort per Kuppermann rule derivation.

    Default: GCS 14-15, and either CT was done (so we have PosCT) or we have
    outcome (e.g. PosIntFinal, DeathTBI) for those not imaged.
    """
    mask = (df["GCSTotal"] >= gcs_range[0]) & (df["GCSTotal"] <= gcs_range[1])
    if require_ct_or_outcome:
        has_ct = df["CTDone"] == 1
        has_outcome = df["PosIntFinal"].notna() | df["DeathTBI"].notna()
        mask = mask & (has_ct | has_outcome)
    return df[mask].copy()


def load_and_clean(
    data_path: str,
    na_values: list[str] | None = None,
) -> pd.DataFrame:
    """Load CSV and apply clean_data."""
    if na_values is None:
        na_values = ["", "NA", "N/A", "Unknown"]
    df = pd.read_csv(data_path, low_memory=False, na_values=na_values)
    return clean_data(df)


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    csv_path = data_dir / "TBI PUD 10-08-2013.csv"
    if not csv_path.exists():
        print("Data file not found at", csv_path)
    else:
        df = load_and_clean(str(csv_path))
        df = flag_inconsistencies(df)
        print("Loaded and cleaned:", df.shape[0], "rows,", df.shape[1], "columns")
        print("GCS mismatches:", df["GCS_mismatch"].sum())
        print("Age mismatches:", df["Age_mismatch"].sum())
        df["ciTBI"] = derive_ciTBI(df)
        cohort = get_analysis_cohort(df)
        print("Analysis cohort (GCS 14-15):", len(cohort), "rows")
