import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter, sobel as _sobel_filter

def apply_feat_all_images(dfs, feat, feat_name, *args, **kwargs):
    """Apply a feature function to every DataFrame in dfs, adding a new column."""
    dfs_with_features = dfs.copy()
    for name, df in dfs_with_features.items():
        df[feat_name] = feat(df, *args, **kwargs)
        dfs_with_features[name] = df
    return dfs_with_features


def rank_difference(df, dist='l2', rank='standard'):
    """Rank difference across radiance angles using l1 or l2 distance."""
    radiances = ['Rad_DF', 'Rad_CF', 'Rad_BF', 'Rad_AF', 'Rad_AN']
    radiance_ranks = [s + '_rank' for s in radiances]
    new_df = df.copy()

    for angle in radiances:
        if rank == 'standard':
            new_df[angle + '_rank'] = df[angle].rank(ascending=False, method='average')
        elif rank == 'z-score':
            new_df[angle + '_rank'] = stats.zscore(df[angle], nan_policy='omit')

    rank_vals = new_df.loc[:, radiance_ranks].values
    means = rank_vals.mean(axis=1)
    diffs = rank_vals - means[:, np.newaxis]
    
    if (dist == 'l1'):
        mean_rank_diff = np.mean(np.abs(diffs), axis=1)
    elif (dist == 'l2'):
        mean_rank_diff = np.mean(diffs ** 2, axis=1)
    else:
        raise ValueError("Need either l1 or l2 as distance metrics.")

    return mean_rank_diff

def pooled_rank_difference(df, patch_size=9, dist='l2', rank='standard'):
    """Local pooling of rank_difference over spatial patches."""
    rank_diff = rank_difference(df, dist=dist, rank=rank)
    rank_diff = (rank_diff - rank_diff.mean()) / (rank_diff.std() + 1e-8)

    y_coords = df['y'].astype(int).values
    x_coords = df['x'].astype(int).values
    y_shifted = y_coords - y_coords.min()
    x_shifted = x_coords - x_coords.min()
    max_y = y_shifted.max() + 1
    max_x = x_shifted.max() + 1

    valid = np.zeros((max_y, max_x), dtype=np.float64)
    grid  = np.zeros((max_y, max_x), dtype=np.float64)
    valid[y_shifted, x_shifted] = 1.0
    grid[y_shifted, x_shifted]  = rank_diff

    local_mean  = uniform_filter(grid,      size=patch_size, mode='reflect')
    local_mean2 = uniform_filter(grid ** 2, size=patch_size, mode='reflect')

    if dist == 'l2':
        result_grid = np.clip(local_mean2 - local_mean ** 2, 0, None)
    else:
        result_grid = np.sqrt(np.clip(local_mean2 - local_mean ** 2, 0, None))

    return result_grid[y_shifted, x_shifted]

def pooled_feature(df, feat, transform='mean', patch_size=9):
    """Pool a feature over local spatial patches (mean, median, or sd)."""
    new_df = df.copy()

    y_coords = df['y'].astype(int).values
    x_coords = df['x'].astype(int).values
    y_min_coord = y_coords.min()
    x_min_coord = x_coords.min()
    y_shifted = y_coords - y_min_coord
    x_shifted = x_coords - x_min_coord
    max_y = y_shifted.max() + 1
    max_x = x_shifted.max() + 1

    valid = np.zeros((max_y, max_x), dtype=np.float64)
    grid = np.zeros((max_y, max_x), dtype=np.float64)
    valid[y_shifted, x_shifted] = 1.0
    grid[y_shifted, x_shifted] = new_df[feat].values

    if transform == 'mean':
        result_grid = uniform_filter(grid, size=patch_size, mode='reflect')
    elif transform == 'sd':
        local_mean  = uniform_filter(grid,      size=patch_size, mode='reflect')
        local_mean2 = uniform_filter(grid ** 2, size=patch_size, mode='reflect')
        result_grid = np.sqrt(np.clip(local_mean2 - local_mean ** 2, 0, None))
    elif transform == 'median':
        from scipy.ndimage import generic_filter
        filled = np.where(valid > 0, grid, np.nan)
        result_grid = generic_filter(
            filled, lambda x: np.nanmedian(x), size=patch_size, mode='reflect'
        )
    else:
        raise ValueError("Please select transformation from: mean/median/sd")

    return result_grid[y_shifted, x_shifted]

def create_many_transforms(dfs, feats_to_try=['NDAI', 'SD', 'CORR'], transforms=['mean', 'sd'], patch_size=9):
    """Apply multiple pooling transforms to multiple features across all images."""
    new_dfs = dfs.copy()
    for name, df in new_dfs.items():
        new_df = df.copy()
        for feat in feats_to_try:
            for transform in transforms:
                new_feat_name = f"{feat}_{transform}_patch{patch_size}"
                new_df[new_feat_name] = pooled_feature(df, feat, transform=transform, patch_size=patch_size)
        new_dfs[name] = new_df
    return new_dfs

def sobel_gradient(df, feat):
    """Sobel gradient magnitude for a feature on the spatial grid."""
    new_df = df.copy()

    y_coords = df['y'].astype(int).values
    x_coords = df['x'].astype(int).values
    y_min_coord = y_coords.min()
    x_min_coord = x_coords.min()
    y_shifted = y_coords - y_min_coord
    x_shifted = x_coords - x_min_coord
    max_y = y_shifted.max() + 1
    max_x = x_shifted.max() + 1

    grid = np.zeros((max_y, max_x), dtype=np.float64)
    grid[y_shifted, x_shifted] = new_df[feat].values

    gx = _sobel_filter(grid, axis=1, mode='reflect')
    gy = _sobel_filter(grid, axis=0, mode='reflect')
    gradient_grid = np.abs(gx) + np.abs(gy)

    std = gradient_grid[y_shifted, x_shifted].std()
    if std > 0:
        gradient_grid = gradient_grid / std

    return gradient_grid[y_shifted, x_shifted]

def build_features(df):
    """
    Builds the additional features for a given df.
    """
    EPS = 1e-6
    RADIANCE_NAMES = ["Rad_DF", "Rad_CF", "Rad_BF", "Rad_AF", "Rad_AN"]
    
    base = df.copy()

    ndai, sd, corr = base.loc[:, ['NDAI']], base.loc[:, ['SD']], base.loc[:, ['CORR']]
    rad = base.loc[:, RADIANCE_NAMES]

    sd_log = np.log1p(np.clip(sd, 0.0, None))
    base[['SD_log']] = sd_log
    base[[f"log_{n}" for n in RADIANCE_NAMES]] = np.log1p(np.clip(rad, 0.0, None))

    # Angular anisotropy: normalised difference vs nadir-like AN angle
    an = rad.iloc[:, 4:5]

    base[["aniso_DF_AN", "aniso_CF_AN", "aniso_BF_AN", "aniso_AF_AN"]] = (rad.iloc[:, :4].values - an.values) / (rad.iloc[:, :4].values + an.values + EPS)

    base[["ratio_DF_CF", "ratio_CF_BF", "ratio_BF_AF", "ratio_AF_AN"]] = np.concatenate([
        rad.iloc[:, 0:1].values / (rad.iloc[:, 1:2].values + EPS),
        rad.iloc[:, 1:2].values / (rad.iloc[:, 2:3].values + EPS),
        rad.iloc[:, 2:3].values / (rad.iloc[:, 3:4].values + EPS),
        rad.iloc[:, 3:4].values / (rad.iloc[:, 4:5].values + EPS),
    ], axis=1)

    base[["NDAI_x_CORR", "NDAI_x_SDlog", "CORR_x_SDlog"]] = np.concatenate([ndai.values * corr.values, ndai.values * sd_log.values, corr.values * sd_log.values], axis=1)
    
    return base

def build_features_for_all_dfs(dfs):
    """
    Applies the build_features function to all DataFrames in the input dictionary.
    """
    new_dfs = dfs.copy()
    for name, df in new_dfs.items():
        new_df = df.copy()
        new_dfs[name] = build_features(new_df)
    return new_dfs