"""
Comprehensive data cleaning and exploratory analysis for PECARN TBI data.
Generates diagnostic summaries and figures for the lab report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean import (
    load_and_clean,
    clean_data,
    flag_inconsistencies,
    derive_ciTBI,
    get_analysis_cohort,
)


def analyze_missingness(df: pd.DataFrame, out_path: Path | None = None) -> pd.DataFrame:
    """Summarize missingness patterns and return a report DataFrame."""
    n = len(df)
    missing = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct = n_miss / n * 100
        missing.append({"variable": col, "n_missing": n_miss, "pct_missing": pct})

    report = pd.DataFrame(missing).sort_values("pct_missing", ascending=False)
    if out_path:
        report.head(40).to_csv(out_path, index=False)
    return report


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Check for data quality issues: GCS, age, outcome consistency."""
    issues = {}

    # GCS component sum vs total
    if all(c in df.columns for c in ["GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal"]):
        gcs_sum = df["GCSEye"] + df["GCSVerbal"] + df["GCSMotor"]
        valid = df["GCSTotal"].notna() & df["GCSEye"].notna()
        n_mismatch = (valid & (gcs_sum != df["GCSTotal"])).sum()
        issues["GCS_component_mismatch"] = {
            "n": int(n_mismatch),
            "pct": n_mismatch / valid.sum() * 100 if valid.sum() > 0 else 0,
        }

    # PosCT/CTDone consistency: when CTDone=0, PosCT should be missing
    if "CTDone" in df.columns and "PosCT" in df.columns:
        no_ct = df["CTDone"] == 0
        posct_has_val = df["PosCT"].notna()
        n_inconsistent = (no_ct & posct_has_val).sum()
        issues["PosCT_when_no_CT"] = {"n": int(n_inconsistent)}

    # Age sanity
    if "AgeInMonth" in df.columns:
        valid_age = df["AgeInMonth"].dropna()
        valid_age = valid_age[(valid_age >= 0) & (valid_age <= 252)]  # 0-21 years
        issues["AgeInMonth_range"] = {
            "min": float(valid_age.min()),
            "max": float(valid_age.max()),
            "n_valid": len(valid_age),
        }

    return issues


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "TBI PUD 10-08-2013.csv"
    output_dir = PROJECT_ROOT / "report" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PECARN TBI Data Cleaning and Exploratory Analysis")
    print("=" * 60)

    # 1. Load and clean
    df_raw = pd.read_csv(data_path, low_memory=False, na_values=["", "NA", "N/A"])
    df = load_and_clean(str(data_path))
    print(f"\n1. Loaded {len(df)} rows, {len(df.columns)} columns")
    print("   Replaced 91, 92, 99, 90, -1 with NaN per documentation")

    # 2. Consistency checks
    df = flag_inconsistencies(df)
    n_gcs = df["GCS_mismatch"].sum()
    n_age = df["Age_mismatch"].sum()
    print(f"\n2. Consistency flags:")
    print(f"   GCS component sum != GCSTotal: {n_gcs} rows")
    print(f"   AgeTwoPlus vs AgeInMonth: {n_age} rows")

    # 3. Data quality report
    quality = analyze_data_quality(df)
    print("\n3. Data quality:")
    for k, v in quality.items():
        print(f"   {k}: {v}")

    # 4. Missingness
    missing_report = analyze_missingness(df, output_dir / "missingness_summary.csv")
    print("\n4. Top 10 variables by missingness:")
    for _, row in missing_report.head(10).iterrows():
        print(f"   {row['variable']}: {row['pct_missing']:.1f}%")

    # 5. Derive ciTBI and analysis cohort
    df["ciTBI"] = derive_ciTBI(df)
    cohort = get_analysis_cohort(df)
    print(f"\n5. Analysis cohort (GCS 14-15): {len(cohort)} rows")
    print(f"   ciTBI rate in cohort: {cohort['ciTBI'].mean()*100:.2f}%")

    # 6. Key variable distributions
    print("\n6. Key variable distributions (cleaned):")
    key_vars = ["GCSTotal", "AgeinYears", "Vomit", "LOCSeparate", "HASeverity", "AMS"]
    for v in key_vars:
        if v in df.columns:
            valid = df[v].dropna()
            print(f"   {v}: n={len(valid)}, mean={valid.mean():.2f}, unique={valid.nunique()}")

    # 7. Figures
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 7a. Missingness bar chart (top 20)
    top_miss = missing_report.head(20)
    axes[0, 0].barh(range(len(top_miss)), top_miss["pct_missing"], color="steelblue", alpha=0.8)
    axes[0, 0].set_yticks(range(len(top_miss)))
    axes[0, 0].set_yticklabels(top_miss["variable"], fontsize=8)
    axes[0, 0].set_xlabel("% Missing")
    axes[0, 0].set_title("Missingness by Variable (Top 20)")
    axes[0, 0].invert_yaxis()

    # 7b. Age distribution
    age = cohort["AgeinYears"].dropna()
    axes[0, 1].hist(age, bins=18, edgecolor="white", color="steelblue", alpha=0.8)
    axes[0, 1].set_xlabel("Age (years)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Age Distribution (Analysis Cohort)")

    # 7c. GCS distribution
    gcs = cohort["GCSTotal"].dropna()
    gcs.value_counts().sort_index().plot(kind="bar", ax=axes[1, 0], color="steelblue", alpha=0.8)
    axes[1, 0].set_xlabel("GCS Total")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("GCS Distribution (Analysis Cohort)")

    # 7d. Outcome: PosCT and ciTBI (among those with CT)
    ct_done = cohort[cohort["CTDone"] == 1]
    outcomes = pd.DataFrame({
        "PosCT": ct_done["PosCT"].value_counts(),
        "ciTBI": cohort["ciTBI"].value_counts(),
    })
    outcomes.plot(kind="bar", ax=axes[1, 1], color=["#2ecc71", "#e74c3c"], alpha=0.8)
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Outcome Distribution")
    axes[1, 1].legend(title="Variable")

    plt.tight_layout()
    plt.savefig(output_dir / "eda_overview.pdf")
    plt.close()
    print(f"\n7. Saved {output_dir / 'eda_overview.pdf'}")

    # 7e. PosCT rate by Vomit (key PECARN predictor)
    ct_done = cohort[cohort["CTDone"] == 1]
    if len(ct_done) > 0 and "PosCT" in ct_done.columns and "Vomit" in ct_done.columns:
        fig2, ax = plt.subplots(figsize=(5, 4))
        g = ct_done.groupby("Vomit")["PosCT"].agg(["mean", "count"])
        g = g[g["count"] > 0]
        g["mean"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
        ax.set_xlabel("Vomiting (0=No, 1=Yes)")
        ax.set_ylabel("Positive CT rate")
        ax.set_title("Positive CT rate by vomiting status (among those imaged)")
        ax.set_xticklabels(["No", "Yes"], rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "outcome_by_vomit.pdf")
        plt.close()
        print(f"    Saved {output_dir / 'outcome_by_vomit.pdf'}")

    # 7f. Injury severity vs PosCT rate - line plot (dose-response trend)
    if "High_impact_InjSev" in ct_done.columns:
        sev_labels = {1.0: "Low", 2.0: "Moderate", 3.0: "High"}
        g = ct_done.groupby("High_impact_InjSev")["PosCT"].agg(["mean", "count"])
        g = g[g["count"] > 50].sort_index()
        g.index = [sev_labels.get(i, str(i)) for i in g.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(range(len(g)), g["mean"] * 100, "o-", color="#2c3e50", linewidth=2, markersize=10)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index)
        ax.set_xlabel("Injury mechanism severity")
        ax.set_ylabel("Positive CT rate (%)")
        ax.set_title("Dose-response: PosCT rate by injury severity")
        ax.set_ylim(0, None)
        plt.tight_layout()
        plt.savefig(output_dir / "injury_severity_posct.pdf")
        plt.close()
        print(f"    Saved {output_dir / 'injury_severity_posct.pdf'}")

    # 7f2. Correlation heatmap of PECARN variables (relationships)
    pecarn_numeric = [
        "GCSTotal", "AMS", "Vomit", "LOCSeparate", "High_impact_InjSev",
        "HASeverity", "Hema", "ActNorm", "AgeTwoPlus", "AgeinYears",
    ]
    avail_num = [v for v in pecarn_numeric if v in cohort.columns]
    corr = cohort[avail_num].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(avail_num)))
    ax.set_yticks(range(len(avail_num)))
    ax.set_xticklabels(avail_num, rotation=45, ha="right")
    ax.set_yticklabels(avail_num)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation among PECARN variables")
    plt.tight_layout()
    plt.savefig(output_dir / "pecarn_correlation.pdf")
    plt.close()
    print(f"    Saved {output_dir / 'pecarn_correlation.pdf'}")

    # 7g. AMS vs PosCT rate
    if "AMS" in ct_done.columns:
        g = ct_done.groupby("AMS")["PosCT"].agg(["mean", "count"])
        g = g[g["count"] > 50]
        fig, ax = plt.subplots(figsize=(4, 4))
        g["mean"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
        ax.set_xlabel("Altered mental status")
        ax.set_ylabel("Positive CT rate")
        ax.set_title("Positive CT rate by AMS status")
        ax.set_xticklabels(["No", "Yes"], rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "ams_posct.pdf")
        plt.close()
        print(f"    Saved {output_dir / 'ams_posct.pdf'}")

    # 7h. Age group (AgeTwoPlus) vs PosCT rate
    if "AgeTwoPlus" in ct_done.columns:
        g = ct_done.groupby("AgeTwoPlus")["PosCT"].agg(["mean", "count"])
        g.index = ["< 2 years", ">= 2 years"]
        fig, ax = plt.subplots(figsize=(5, 4))
        g["mean"].plot(kind="bar", ax=ax, color=["#9b59b6", "#3498db"], alpha=0.8)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Positive CT rate")
        ax.set_title("Positive CT rate by age group")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "age_group_posct.pdf")
        plt.close()
        print(f"    Saved {output_dir / 'age_group_posct.pdf'}")

    # 7i. PECARN variables missingness
    pecarn_vars = [
        "GCSTotal", "AMS", "SFxPalp", "Vomit", "LOCSeparate", "LocLen",
        "High_impact_InjSev", "HASeverity", "Hema", "HemaLoc", "ActNorm",
        "SFxBas", "AgeTwoPlus", "AgeinYears",
    ]
    avail = [v for v in pecarn_vars if v in cohort.columns]
    miss_pct = cohort[avail].isna().sum() / len(cohort) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    miss_pct.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
    ax.set_xlabel("% Missing in analysis cohort")
    ax.set_title("Missingness of PECARN rule variables")
    plt.tight_layout()
    plt.savefig(output_dir / "pecarn_missingness.pdf")
    plt.close()
    print(f"    Saved {output_dir / 'pecarn_missingness.pdf'}")

    # 7j. GCS consistency check
    n_match = (~df["GCS_mismatch"]).sum()
    n_mismatch = df["GCS_mismatch"].sum()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Consistent", "Mismatch"], [n_match, n_mismatch], color=["#27ae60", "#e74c3c"], alpha=0.8)
    ax.set_ylabel("Number of rows")
    ax.set_title("GCS: Component sum vs GCSTotal")
    plt.tight_layout()
    plt.savefig(output_dir / "gcs_consistency.pdf")
    plt.close()
    print(f"    Saved {output_dir / 'gcs_consistency.pdf'}")

    # 8. PECARN-relevant variables: missingness in analysis cohort
    pecarn_vars = [
        "GCSTotal", "AMS", "SFxPalp", "Vomit", "LOCSeparate", "LocLen",
        "High_impact_InjSev", "HASeverity", "Hema", "HemaLoc", "ActNorm",
        "SFxBas", "AgeTwoPlus", "AgeinYears",
    ]
    avail = [v for v in pecarn_vars if v in cohort.columns]
    miss_pecarn = cohort[avail].isna().sum() / len(cohort) * 100
    print("\n8. PECARN variables missingness in cohort:")
    for v in avail:
        print(f"   {v}: {miss_pecarn[v]:.1f}%")

    # 9. Outcome by key predictors (CT-positive among those who had CT)
    ct_done = cohort[cohort["CTDone"] == 1]
    if len(ct_done) > 0 and "PosCT" in ct_done.columns:
        print("\n9. PosCT rate by PECARN predictors (among those with CT):")
        for v in ["Vomit", "LOCSeparate", "High_impact_InjSev", "AMS", "AgeTwoPlus"]:
            if v in ct_done.columns and ct_done[v].notna().sum() > 100:
                rates = ct_done.groupby(v)["PosCT"].agg(["mean", "count"])
                print(f"   {v}:")
                for idx, row in rates.iterrows():
                    print(f"      {idx}: PosCT rate={row['mean']*100:.1f}%, n={int(row['count'])}")

    # 10. Outlier check: AgeInMonth
    age = df["AgeInMonth"].dropna()
    age = age[(age >= 0) & (age <= 252)]
    q1, q99 = age.quantile([0.01, 0.99])
    n_out = ((df["AgeInMonth"] < q1) | (df["AgeInMonth"] > q99)).sum()
    print(f"\n10. AgeInMonth outliers (outside 1-99%): {n_out} rows")

    # 11. Save cohort for modeling
    cohort_path = output_dir.parent / "analysis_cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    print(f"\n11. Saved analysis cohort to {cohort_path}")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
