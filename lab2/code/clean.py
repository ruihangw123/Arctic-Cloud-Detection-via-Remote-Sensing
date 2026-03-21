import numpy as np
import pandas as pd
import os

DATA_DIR = "../data"

COLUMN_NAMES = ['y', 'x', 'NDAI', 'SD', 'CORR',
                'Rad_DF', 'Rad_CF', 'Rad_BF', 'Rad_AF', 'Rad_AN', 'label']
FEATURE_COLS = ['NDAI', 'SD', 'CORR', 'Rad_DF', 'Rad_CF', 'Rad_BF', 'Rad_AF', 'Rad_AN']
LABELED_FILES = ['O012791.npz', 'O013257.npz', 'O013490.npz']

def read_data():
    """Load labeled .npz files into a dict of DataFrames keyed by image name."""
    dfs = {}
    for fname in LABELED_FILES:
        fp = os.path.join(DATA_DIR, fname)
        npz = np.load(fp)
        key = list(npz.files)[0]
        arr = npz[key]
        df = pd.DataFrame(arr, columns=COLUMN_NAMES)
        img_name = fname.replace('.npz', '')
        dfs[img_name] = df

    return dfs

def remove_unlabeled(dfs, verbose=False):
    """Drop unlabeled rows and convert label to binary 'cloud' column."""
    dfs_labeled = {}
    for name, df in dfs.items():
        df_clean = df[df['label'] != 0].copy()

        df_clean['cloud'] = (df_clean['label'] == 1).astype(int)
        dfs_labeled[name] = df_clean

        if verbose:
            removed = len(df) - len(df_clean)
            print(f"{name}: {len(df)} -> {len(df_clean)} pixels "
                f"(removed {removed} unlabeled, {100*removed/len(df):.1f}%)")
            print(f"  Cloud: {df_clean['cloud'].sum()} ({100*df_clean['cloud'].mean():.1f}%), "
                f"Not cloud: {(df_clean['cloud']==0).sum()} ({100*(1-df_clean['cloud'].mean()):.1f}%)")


    return dfs_labeled

def combine_labeled(dfs_labeled):
    """Concat all image DataFrames into one, with an 'image' column."""
    all_labeled_raw = pd.concat(dfs_labeled.values(), keys=dfs_labeled.keys(), names=['image'])
    all_labeled_raw = all_labeled_raw.reset_index(level='image')


    return all_labeled_raw

def check_data_quality(dfs, verbose=True):
    """Check for NaN, Inf, negative radiance, duplicate coords, zero rows."""
    radiance_cols = ['Rad_DF', 'Rad_CF', 'Rad_BF', 'Rad_AF', 'Rad_AN']
    report = {}
    for name, df in dfs.items():
        n_nan      = df[FEATURE_COLS].isna().any(axis=1).sum()
        n_inf      = np.isinf(df[FEATURE_COLS].values).any(axis=1).sum()
        n_neg_rad  = (df[radiance_cols] < 0).any(axis=1).sum()
        n_dup_coords = df.duplicated(subset=['y', 'x']).sum()
        n_zero_rows  = (df[FEATURE_COLS] == 0).all(axis=1).sum()
        report[name] = {
            'n_rows':        len(df),
            'n_nan':         int(n_nan),
            'n_inf':         int(n_inf),
            'n_neg_rad':     int(n_neg_rad),
            'n_dup_coords':  int(n_dup_coords),
            'n_zero_rows':   int(n_zero_rows),
        }
        if verbose:
            print(f"{name}: {len(df)} rows | NaN={n_nan} | Inf={n_inf} | "
                  f"neg_rad={n_neg_rad} | dup_coords={n_dup_coords} | zero_rows={n_zero_rows}")
    return report


def remove_bad_rows(dfs, remove_nan=True, remove_inf=True,
                    remove_negative_radiance=False, remove_dup_coords=False,
                    verbose=False):
    """Remove bad rows (NaN, Inf, etc.) per the specified flags."""
    radiance_cols = ['Rad_DF', 'Rad_CF', 'Rad_BF', 'Rad_AF', 'Rad_AN']
    dfs_clean = {}
    for name, df in dfs.items():
        mask = pd.Series(True, index=df.index)
        if remove_nan:
            mask &= ~df[FEATURE_COLS].isna().any(axis=1)
        if remove_inf:
            mask &= ~pd.DataFrame(np.isinf(df[FEATURE_COLS].values),
                                   columns=FEATURE_COLS, index=df.index).any(axis=1)
        if remove_negative_radiance:
            mask &= ~(df[radiance_cols] < 0).any(axis=1)
        if remove_dup_coords:
            mask &= ~df.duplicated(subset=['y', 'x'], keep='first')
        removed = (~mask).sum()
        dfs_clean[name] = df[mask].copy()
        if verbose and removed > 0:
            print(f"{name}: removed {removed} bad rows ({100*removed/len(df):.2f}%)")
    return dfs_clean


def winsorize(dfs_labeled, feature_cols=FEATURE_COLS, lower_percentile=0.01, upper_percentile=0.99, verbose=False):
    """Clip feature columns to the given percentile bounds."""
    dfs_winsorized = dfs_labeled.copy()
    for name, df in dfs_winsorized.items():
        for feat in feature_cols:
            p_low = df[feat].quantile(lower_percentile)
            p_high = df[feat].quantile(upper_percentile)
            n_clipped = ((df[feat] < p_low) | (df[feat] > p_high)).sum()
            df[feat] = df[feat].clip(lower=p_low, upper=p_high)
            if n_clipped > 0 and verbose:
                print(f"  {name} | {feat:>8}: clipped {n_clipped} values to [{p_low:.3f}, {p_high:.3f}]")
        dfs_winsorized[name] = df

    return dfs_winsorized

def clean_data(dfs, do_remove_unlabeled=True, do_winsorize=True,
               winsor_lower=0.01, winsor_upper=0.99,
               remove_negative_radiance=False, remove_dup_coords=False,
               verbose=False):
    """Full cleaning pipeline with toggleable steps for stability analysis."""
    dfs_cleaned = remove_bad_rows(
        dfs,
        remove_negative_radiance=remove_negative_radiance,
        remove_dup_coords=remove_dup_coords,
        verbose=verbose,
    )

    if do_remove_unlabeled:
        dfs_cleaned = remove_unlabeled(dfs_cleaned, verbose=verbose)

    if do_winsorize:
        dfs_cleaned = winsorize(
            dfs_cleaned,
            lower_percentile=winsor_lower,
            upper_percentile=winsor_upper,
            verbose=verbose,
        )

    return dfs_cleaned
