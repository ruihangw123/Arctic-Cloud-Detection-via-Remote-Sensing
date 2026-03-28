import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats

###############################################################################
# Load labeled data (same pattern as generate_eda_figs.py)
###############################################################################

col_names = ['y', 'x', 'ndai', 'sd', 'corr',
             'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an',
             'cloud']

labeled_image_1 = pd.DataFrame(
    np.load("../data/image_data/O013257.npz")['arr_0'], columns=col_names)
labeled_image_2 = pd.DataFrame(
    np.load("../data/image_data/O013490.npz")['arr_0'], columns=col_names)
labeled_image_3 = pd.DataFrame(
    np.load("../data/image_data/O012791.npz")['arr_0'], columns=col_names)

labeled_image_1['image'] = 'O013257'
labeled_image_2['image'] = 'O013490'
labeled_image_3['image'] = 'O012791'

all_labeled = pd.concat(
    [labeled_image_1, labeled_image_2, labeled_image_3], ignore_index=True)

# Filter to only labeled pixels (cloud=1 or cloud=-1), drop unlabeled (0)
df = all_labeled[all_labeled['cloud'] != 0].copy()
# Convert labels for sklearn
# -1 -> 0 (not cloud)
# +1 -> 1 (cloud)
df['label'] = (df['cloud'] == 1).astype(int)

feature_cols = ['ndai', 'sd', 'corr',
                'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an']


###############################################################################
# 1. Per-feature AUC 
# Measures how well each feature alone separates the two classes
###############################################################################

print("=" * 60)
print("PER-FEATURE AUC")
print("=" * 60)

auc_results = {}
for feat in feature_cols:
    vals = df[feat].values
    labels = df['label'].values
    # Handle the case where lower values might indicate cloud
    auc = roc_auc_score(labels, vals)
    # If AUC < 0.5, the feature is inversely related, so we should flip it
    auc_results[feat] = max(auc, 1 - auc)
    direction = "higher = cloud" if auc >= 0.5 else "lower = cloud"
    print(f"  {feat:12s}: AUC = {auc_results[feat]:.4f}  ({direction})")

print()


###############################################################################
# 2. Mutual Information
# Measures nonlinear association with label
###############################################################################

print("=" * 60)
print("MUTUAL INFORMATION (discretized, nonlinear dependence)")
print("=" * 60)

mi_results = {}
for feat in feature_cols:
    # Discretize feature into 20 bins for easier computation
    discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    feat_binned = discretizer.fit_transform(df[[feat]]).ravel().astype(int)
    mi = mutual_info_score(df['label'].values, feat_binned)
    mi_results[feat] = mi
    print(f"  {feat:12s}: MI = {mi:.4f}")

print()


###############################################################################
# 3. t-statistics
# Measures the difference in means between classes
###############################################################################

print("=" * 60)
print("|t-statistic|")
print("=" * 60)

ttest_results = {}
for feat in feature_cols:
    cloud_vals = df.loc[df['label'] == 1, feat].values
    nocloud_vals = df.loc[df['label'] == 0, feat].values
    t_stat, p_val = stats.ttest_ind(cloud_vals, nocloud_vals, equal_var=False)
    ttest_results[feat] = abs(t_stat)
    print(f"  {feat:12s}: |t| = {abs(t_stat):.1f},  p = {p_val:.2e}")

print()


###############################################################################
# 4. Per-image AUC stability check
###############################################################################

print("=" * 60)
print("PER-IMAGE AUC (stability across images)")
print("=" * 60)

image_auc = {}
for img_name in ['O013257', 'O013490', 'O012791']:
    img_df = df[df['image'] == img_name]
    image_auc[img_name] = {}
    for feat in feature_cols:
        auc = roc_auc_score(img_df['label'].values, img_df[feat].values)
        image_auc[img_name][feat] = max(auc, 1 - auc)

print(f"  {'Feature':12s}  {'O013257':>8s}  {'O013490':>8s}  {'O012791':>8s}  {'Std':>6s}")
print("  " + "-" * 50)
for feat in feature_cols:
    vals = [image_auc[img][feat] for img in ['O013257', 'O013490', 'O012791']]
    std = np.std(vals)
    print(f"  {feat:12s}  {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}  {std:6.4f}")

print()


###############################################################################
# SUMMARY: Rank features
###############################################################################

print("=" * 60)
print("OVERALL FEATURE RANKING")
print("=" * 60)

# Create summary dataframe
summary = pd.DataFrame({
    'AUC': auc_results,
    'MI': mi_results,
    '|t|': ttest_results,
})

# Rank features for each metric
for col in summary.columns:
    summary[f'{col}_rank'] = summary[col].rank(ascending=False).astype(int)

summary['avg_rank'] = summary[['AUC_rank', 'MI_rank', '|t|_rank']].mean(axis=1)
summary = summary.sort_values('avg_rank')

# Identify top 3 features
print(summary[['AUC', 'MI', '|t|', 'avg_rank']].to_string())
print()
print("Top 3 features:", list(summary.index[:3]))


###############################################################################
# FIGURE 1: Feature importance bar chart
###############################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AUC bars
auc_sorted = dict(sorted(auc_results.items(), key=lambda x: x[1], reverse=True))
colors = ['#2196F3' if i < 3 else '#BBDEFB' for i in range(len(auc_sorted))]
axes[0].barh(list(auc_sorted.keys()), list(auc_sorted.values()), color=colors)
axes[0].set_xlabel('AUC')
axes[0].set_title('Per-Feature AUC')
axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlim(0.45, 1.0)

# MI bars
mi_sorted = dict(sorted(mi_results.items(), key=lambda x: x[1], reverse=True))
colors = ['#4CAF50' if i < 3 else '#C8E6C9' for i in range(len(mi_sorted))]
axes[1].barh(list(mi_sorted.keys()), list(mi_sorted.values()), color=colors)
axes[1].set_xlabel('Mutual Information')
axes[1].set_title('Mutual Information with Label')

# t-test bars
t_sorted = dict(sorted(ttest_results.items(), key=lambda x: x[1], reverse=True))
colors = ['#FF9800' if i < 3 else '#FFE0B2' for i in range(len(t_sorted))]
axes[2].barh(list(t_sorted.keys()), list(t_sorted.values()), color=colors)
axes[2].set_xlabel('|t-statistic|')
axes[2].set_title("Welch's t-test |t-statistic|")

plt.suptitle('Feature Importance: Three Quantitative Metrics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../figs/feature_importance_bars.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/feature_importance_bars.png")


###############################################################################
# FIGURE 2: Per-image AUC stability heatmap
###############################################################################

stability_df = pd.DataFrame(image_auc).T
stability_df.columns.name = 'Feature'
stability_df.index.name = 'Image'

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(stability_df, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
ax.set_title('Per-Feature AUC by Image (Stability Check)')
plt.tight_layout()
plt.savefig("../figs/feature_importance_stability.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/feature_importance_stability.png")


###############################################################################
# FIGURE 3: Density plots for top 3 features by class
###############################################################################

top3 = list(summary.index[:3])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feat in enumerate(top3):
    cloud_vals = df.loc[df['label'] == 1, feat]
    nocloud_vals = df.loc[df['label'] == 0, feat]
    axes[i].hist(nocloud_vals, bins=80, density=True, alpha=0.6,
                 color='lightsalmon', label='Not Cloud', edgecolor='none')
    axes[i].hist(cloud_vals, bins=80, density=True, alpha=0.6,
                 color='skyblue', label='Cloud', edgecolor='none')
    axes[i].set_xlabel(feat.upper())
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'{feat.upper()} Distribution by Class')
    axes[i].legend()

plt.suptitle('Top 3 Features: Class-Conditional Density', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../figs/feature_importance_density.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: ../figs/feature_importance_density.png")