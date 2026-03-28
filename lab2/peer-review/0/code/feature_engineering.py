# Creates new engineered features for the 3 labeled images
# Outputs: enriched CSVs + diagnostic figures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter, sobel
from sklearn.metrics import roc_auc_score

###############################################################################
# Load labeled data
###############################################################################

col_names = ['y', 'x', 'ndai', 'sd', 'corr',
             'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an',
             'cloud']

image_files = {
    'O013257': '../data/image_data/O013257.npz',
    'O013490': '../data/image_data/O013490.npz',
    'O012791': '../data/image_data/O012791.npz',
}

images = {}
for name, path in image_files.items():
    df = pd.DataFrame(np.load(path)['arr_0'], columns=col_names)
    df['image'] = name
    images[name] = df


###############################################################################
# Helper functions
# create a 2D grid from a column, apply a spatial operation, map back
###############################################################################

def column_to_grid(df, col):
    """Pivot a column into a 2D grid using y, x coordinates."""
    grid = df.pivot(index='y', columns='x', values=col)
    return grid.values, grid.index, grid.columns


def grid_to_column(df, grid_values, row_idx, col_idx):
    """Map 2D grid values back to the dataframe rows."""
    # Build a lookup from (y, x) to grid value
    row_map = {v: i for i, v in enumerate(row_idx)}
    col_map = {v: i for i, v in enumerate(col_idx)}
    result = np.zeros(len(df))
    for k in range(len(df)):
        y = df.iloc[k]['y']
        x = df.iloc[k]['x']
        ri = row_map.get(y)
        ci = col_map.get(x)
        if ri is not None and ci is not None:
            result[k] = grid_values[ri, ci]
        else:
            result[k] = np.nan
    return result


def fast_grid_to_column(df, grid_values, row_idx, col_idx):
    """Vectorized version: map 2D grid values back to dataframe rows."""
    row_map = pd.Series(np.arange(len(row_idx)), index=row_idx)
    col_map = pd.Series(np.arange(len(col_idx)), index=col_idx)
    ri = row_map.reindex(df['y'].values).values.astype(int)
    ci = col_map.reindex(df['x'].values).values.astype(int)
    return grid_values[ri, ci]


def apply_spatial_feature(df, col, func, func_name):
    """
    Apply a function to a column via grid representation.
    func takes a 2D array and returns a 2D array of the same shape.
    """
    grid_vals, row_idx, col_idx = column_to_grid(df, col)
    grid_clean = np.nan_to_num(grid_vals, nan=0.0)      # Replace NaN with 0 for spatial operations
    result_grid = func(grid_clean)
    return fast_grid_to_column(df, result_grid, row_idx, col_idx)


###############################################################################
# Define new features
###############################################################################

def engineer_features(df):
    """Add all engineered features to the dataframe."""
    df = df.copy()

    # Feature 1: Radiance ratio Df/An
    # Clouds scatter more in forward direction, resulting in a higher ratio
    df['rad_ratio'] = df['angle_df'] / df['angle_an'].replace(0, np.nan)

    # Feature 2: Radiance range (max - min across all angles)
    # Higher for clouds due to greater scattering variation
    angle_cols = ['angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an']
    df['rad_range'] = df[angle_cols].max(axis=1) - df[angle_cols].min(axis=1)

    # Feature 3: Radiance standard deviation across angles
    # Another measure of variation among angles
    df['rad_std'] = df[angle_cols].std(axis=1)

    # The next 4 features use local spatial statistics in a 5x5 patch
    neighborhood_size = 5

    # Helper functions
    def local_mean(grid):
        """Replace each pixel with the average of its surrounding neighborhood."""
        return uniform_filter(grid.astype(float), size=neighborhood_size, mode='reflect')

    def local_std(grid):
        """Computes local standard deviation in a neighborhood"""
        mean = uniform_filter(grid.astype(float), size=neighborhood_size, mode='reflect')
        mean_sq = uniform_filter(grid.astype(float)**2, size=neighborhood_size, mode='reflect')
        variance = mean_sq - mean**2
        variance = np.maximum(variance, 0)
        return np.sqrt(variance)

    # Feature 4: Local mean of NDAI
    df['ndai_local_mean'] = apply_spatial_feature(df, 'ndai', local_mean, 'ndai_local_mean')

    # Feature 5: Local std of NDAI
    df['ndai_local_std'] = apply_spatial_feature(df, 'ndai', local_std, 'ndai_local_std')

    # Feature 6: Local mean of SD
    df['sd_local_mean'] = apply_spatial_feature(df, 'sd', local_mean, 'sd_local_mean')

    # Feature 7: Local std of SD
    df['sd_local_std'] = apply_spatial_feature(df, 'sd', local_std, 'sd_local_std')

    # Feature 8: Gradient magnitude of NDAI
    # Each pixel gets a value based on how sharply NDAI changes around it
    # This helps us detect edges between clouds and surface
    def gradient_magnitude(grid):
        """Computes a map of edge strengths"""
        gx = sobel(grid.astype(float), axis=1)
        gy = sobel(grid.astype(float), axis=0)
        return np.sqrt(gx**2 + gy**2)

    df['ndai_gradient'] = apply_spatial_feature(df, 'ndai', gradient_magnitude, 'ndai_gradient')

    return df


###############################################################################
# Apply feature engineering to all 3 images
###############################################################################

engineered = {}
for name, df in images.items():
    print(f"Processing image {name}")
    engineered[name] = engineer_features(df)
    print(f"Done processing image {name}")

# Combine all
all_data = pd.concat(engineered.values(), ignore_index=True)


###############################################################################
# Evaluate new features: AUC on labeled pixels
###############################################################################

# Keep only expert labeled pixels
# Convert target to binary: Cloud=1, Not Cloud=0
labeled = all_data[all_data['cloud'] != 0].copy()
labeled['label'] = (labeled['cloud'] == 1).astype(int)

new_features = ['rad_ratio', 'rad_range', 'rad_std',
                'ndai_local_mean', 'ndai_local_std',
                'sd_local_mean', 'sd_local_std',
                'ndai_gradient']

original_features = ['ndai', 'sd', 'corr',
                     'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an']

print("\n" + "=" * 60)
print("NEW FEATURE AUCs vs ORIGINAL FEATURES")
print("=" * 60)

# Evaluate each feature independently with ROC AUC
all_features = original_features + new_features
auc_dict = {}
for feat in all_features:
    vals = labeled[feat].dropna()
    labs = labeled.loc[vals.index, 'label']
    auc = roc_auc_score(labs, vals)
    auc = max(auc, 1 - auc)
    auc_dict[feat] = auc
    marker = " ** NEW" if feat in new_features else ""
    print(f"  {feat:20s}: AUC = {auc:.4f}{marker}")

print()

# Check consistency of each feature for each image
# Lower std acorss images means the feature has more stable behavior
print("=" * 60)
print("NEW FEATURE STABILITY (per-image AUC)")
print("=" * 60)
print(f"  {'Feature':20s}  {'O013257':>8s}  {'O013490':>8s}  {'O012791':>8s}  {'Std':>6s}")
print("  " + "-" * 55)

for feat in new_features:
    aucs = []
    for name in ['O013257', 'O013490', 'O012791']:
        img_df = engineered[name]
        img_labeled = img_df[img_df['cloud'] != 0].copy()
        img_labeled['label'] = (img_labeled['cloud'] == 1).astype(int)
        vals = img_labeled[feat].dropna()
        labs = img_labeled.loc[vals.index, 'label']
        auc = roc_auc_score(labs, vals)
        auc = max(auc, 1 - auc)
        aucs.append(auc)
    std = np.std(aucs)
    print(f"  {feat:20s}  {aucs[0]:8.4f}  {aucs[1]:8.4f}  {aucs[2]:8.4f}  {std:6.4f}")


###############################################################################
# Save data with engineered features
###############################################################################

for name, df in engineered.items():
    out_path = f"../data/image_data/{name}_enriched.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


###############################################################################
# FIGURE 1: AUC comparison bar chart (original vs new features)
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

features_sorted = sorted(auc_dict.keys(), key=lambda x: auc_dict[x])
aucs_sorted = [auc_dict[f] for f in features_sorted]
colors = ['#4CAF50' if f in new_features else '#2196F3' for f in features_sorted]

bars = ax.barh(features_sorted, aucs_sorted, color=colors)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('AUC')
ax.set_title('Feature AUC Comparison: Original (blue) vs Engineered (green)')
ax.set_xlim(0.45, 1.0)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='Original'),
                   Patch(facecolor='#4CAF50', label='Engineered')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig("../figs/feature_engineering_auc.png", dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: ../figs/feature_engineering_auc.png")


###############################################################################
# FIGURE 2: Density plots for best new features
###############################################################################

# Pick top 4 new features by AUC
new_auc = {f: auc_dict[f] for f in new_features}
top_new = sorted(new_auc, key=new_auc.get, reverse=True)[:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, feat in enumerate(top_new):
    ax = axes[i // 2][i % 2]
    cloud_vals = labeled.loc[labeled['label'] == 1, feat].dropna()
    nocloud_vals = labeled.loc[labeled['label'] == 0, feat].dropna()
    ax.hist(nocloud_vals, bins=80, density=True, alpha=0.6,
            color='lightsalmon', label='Not Cloud', edgecolor='none')
    ax.hist(cloud_vals, bins=80, density=True, alpha=0.6,
            color='skyblue', label='Cloud', edgecolor='none')
    ax.set_xlabel(feat)
    ax.set_title(f'{feat} (AUC={auc_dict[feat]:.3f})')
    ax.legend()

plt.suptitle('Top 4 Engineered Features: Class-Conditional Density', fontsize=14)
plt.tight_layout()
plt.savefig("../figs/feature_engineering_density.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/feature_engineering_density.png")


###############################################################################
# FIGURE 3: Spatial visualization of best new feature
###############################################################################

best_new = top_new[0]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Spatial View: {best_new} vs Expert Labels', fontsize=16)

import matplotlib.colors as mcolors
cmap_label = mcolors.ListedColormap(['gray', 'black', 'white'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm_label = mcolors.BoundaryNorm(bounds, cmap_label.N)

for i, name in enumerate(['O013257', 'O013490', 'O012791']):
    df = engineered[name]
    # Expert labels
    grid_label = df.pivot(index='y', columns='x', values='cloud').fillna(0)
    axes[0, i].imshow(grid_label, cmap=cmap_label, norm=norm_label, origin='upper')
    axes[0, i].set_title(f'{name} Expert Labels')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    # New feature
    grid_feat = df.pivot(index='y', columns='x', values=best_new)
    im = axes[1, i].imshow(grid_feat, cmap='viridis', origin='upper')
    axes[1, i].set_title(f'{name} {best_new}')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=best_new)

plt.savefig("../figs/feature_engineering_spatial.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/feature_engineering_spatial.png")
