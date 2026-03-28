# Generates the following EDA plots:

# expert_labels.png
# correlation_heatmap.png
# visual_comparison_radiance_O013257.png
# visual_comparison_radiance_O013490.png
# visual_comparison_radiance_O012791.png
# boxplot_radiance_by_label.png
# boxplot_radiance_by_angle.png
# boxplot_radiance_filtered.png
# boxplot_ndai_filtered.png
# boxplot_sd_filtered.png
# boxplot_corr_filtered.png
# visual_comparison_ndai.png
# visual_comparison_sd.png
# visual_comparison_corr.png

# Saves them all to the /figs folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


###############################################################################
# Load data
###############################################################################

# Load images with expert labels
# 0013257.npz, 0013490.npz, 0012791.npz
expert_labeled_images = ['O013257.npz', 'O013490.npz', 'O012791.npz']
col_names = ['y', 'x', 'ndai', 'sd', 'corr', 'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an', 'cloud']

labeled_image_1 = pd.DataFrame(np.load("../data/image_data/O013257.npz")['arr_0'], columns=col_names)
labeled_image_2 = pd.DataFrame(np.load("../data/image_data/O013490.npz")['arr_0'], columns=col_names)
labeled_image_3 = pd.DataFrame(np.load("../data/image_data/O012791.npz")['arr_0'], columns=col_names)

# Add image identifier column to each dataframe
labeled_image_1['image'] = 'O013257'
labeled_image_2['image'] = 'O013490'
labeled_image_3['image'] = 'O012791'

# Stack dataframes into single dataframe
labeled_images = pd.concat([labeled_image_1, labeled_image_2, labeled_image_3], ignore_index=True)


###############################################################################
# Visualize expert labels
# Save to: "../figs/expert_labels.png"
###############################################################################

# Reshape data into 2D grid for plotting
grid_1 = labeled_image_1.pivot(index='y', columns='x', values='cloud')
grid_2 = labeled_image_2.pivot(index='y', columns='x', values='cloud')
grid_3 = labeled_image_3.pivot(index='y', columns='x', values='cloud')

# Replace NA values with 0 for unlabeled
grid_1 = grid_1.fillna(0)
grid_2 = grid_2.fillna(0)
grid_3 = grid_3.fillna(0)

# Define color map: -1 (not cloud) = gray, 0 (unlabeled) = black, 1 (cloud) = white
cmap = mcolors.ListedColormap(['gray', 'black', 'white'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

ax1.imshow(grid_1, cmap=cmap, norm=norm, origin='upper')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Cloud Expert Labels: O013257.npz')

ax2.imshow(grid_2, cmap=cmap, norm=norm, origin='upper')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Cloud Expert Labels: O013490.npz')

ax3.imshow(grid_3, cmap=cmap, norm=norm, origin='upper')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Cloud Expert Labels: O012791.npz')

# Create legend
legend_patches = [
    Patch(facecolor='gray', edgecolor='gray', label='Not Cloud (-1)'),
    Patch(facecolor='black', edgecolor='black', label='Unlabeled (0)'),
    Patch(facecolor='white', edgecolor='black', label='Cloud (1)')
]

fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.05, 0.5))

plt.tight_layout()

# Save plot
plt.savefig("../figs/expert_labels.png", dpi=300, bbox_inches='tight')


###############################################################################
# Correlation Heatmap
# Save to: "../figs/correlation_heatmap.png"
###############################################################################

# Examine correlation between features
feature_cols = ['ndai', 'sd', 'corr', 'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an', 'cloud']

# All data
corr_matrix_all = labeled_images[feature_cols].corr()

# Excluding unlabeled
labeled_images_filtered = labeled_images[labeled_images['cloud'] != 0]
corr_matrix_labeled = labeled_images_filtered[feature_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# All data
sns.heatmap(
    corr_matrix_all, 
    annot=True, 
    fmt='.2f',
    cmap='RdBu_r', 
    center=0,
    vmin=-1, 
    vmax=1,
    square=True,
    cbar=False,
    ax=axes[0]
)
axes[0].set_title('Correlation Heatmap (All Data)')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

# Excluding unlabeled
sns.heatmap(
    corr_matrix_labeled, 
    annot=True, 
    fmt='.2f',
    cmap='RdBu_r', 
    center=0,
    vmin=-1, 
    vmax=1,
    square=True,
    cbar=True,
    ax=axes[1]
)
axes[1].set_title('Correlation Heatmap (Excluding Unlabeled)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("../figs/correlation_heatmap.png", dpi=300, bbox_inches='tight')


###############################################################################
# Visualize radiances for image O013257 across different angles
# Save to: "../figs/visual_comparison_radiance_O013257.png"
###############################################################################

grid_df_1 = labeled_image_1.pivot(index='y', columns='x', values='angle_df')
grid_cf_1 = labeled_image_1.pivot(index='y', columns='x', values='angle_cf')
grid_bf_1 = labeled_image_1.pivot(index='y', columns='x', values='angle_bf')
grid_af_1 = labeled_image_1.pivot(index='y', columns='x', values='angle_af')
grid_an_1 = labeled_image_1.pivot(index='y', columns='x', values='angle_an')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of Radiance Across Viewing Angles: Image O013257', fontsize=16)

# Image 1 expert labels
axes[0, 0].imshow(grid_1, cmap=cmap, norm=norm, origin='upper')
axes[0, 0].set_title('Expert Cloud Labels')

# Df angle
im1 = axes[0, 1].imshow(grid_df_1, cmap='plasma_r', origin='upper')
axes[0, 1].set_title('Df Angle (70.5°) Radiance')

# Cf angle
im2 = axes[0, 2].imshow(grid_cf_1, cmap='plasma_r', origin='upper')
axes[0, 2].set_title('Cf Angle (60.0°) Radiance')

# Bf angle
im3 = axes[1, 0].imshow(grid_bf_1, cmap='plasma_r', origin='upper')
axes[1, 0].set_title('Bf Angle (45.6°) Radiance')

# Af angle
im4 = axes[1, 1].imshow(grid_af_1, cmap='plasma_r', origin='upper')
axes[1, 1].set_title('Af Angle (26.1°) Radiance')

# An angle
im5 = axes[1, 2].imshow(grid_an_1, cmap='plasma_r', origin='upper')
axes[1, 2].set_title('An Angle (0.0°) Radiance')

# Remove ticks from all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
fig.colorbar(im5, cax=cbar_ax, orientation='horizontal', label='Radiance')

plt.savefig("../figs/visual_comparison_radiance_O013257.png", dpi=300, bbox_inches='tight')


###############################################################################
# Visualize radiances for image O013490 across different angles
# Save to: "../figs/visual_comparison_radiance_O013490.png"
###############################################################################

grid_df_2 = labeled_image_2.pivot(index='y', columns='x', values='angle_df')
grid_cf_2 = labeled_image_2.pivot(index='y', columns='x', values='angle_cf')
grid_bf_2 = labeled_image_2.pivot(index='y', columns='x', values='angle_bf')
grid_af_2 = labeled_image_2.pivot(index='y', columns='x', values='angle_af')
grid_an_2 = labeled_image_2.pivot(index='y', columns='x', values='angle_an')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of Radiance Across Viewing Angles: Image O013490', fontsize=16)

# Image 2 expert labels
axes[0, 0].imshow(grid_2, cmap=cmap, norm=norm, origin='upper')
axes[0, 0].set_title('Expert Cloud Labels')

# Df angle
im1 = axes[0, 1].imshow(grid_df_2, cmap='plasma_r', origin='upper')
axes[0, 1].set_title('Df Angle (70.5°) Radiance')

# Cf angle
im2 = axes[0, 2].imshow(grid_cf_2, cmap='plasma_r', origin='upper')
axes[0, 2].set_title('Cf Angle (60.0°) Radiance')

# Bf angle
im3 = axes[1, 0].imshow(grid_bf_2, cmap='plasma_r', origin='upper')
axes[1, 0].set_title('Bf Angle (45.6°) Radiance')

# Af angle
im4 = axes[1, 1].imshow(grid_af_2, cmap='plasma_r', origin='upper')
axes[1, 1].set_title('Af Angle (26.1°) Radiance')

# An angle
im5 = axes[1, 2].imshow(grid_an_2, cmap='plasma_r', origin='upper')
axes[1, 2].set_title('An Angle (0.0°) Radiance')

# Remove ticks from all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
fig.colorbar(im5, cax=cbar_ax, orientation='horizontal', label='Radiance')

plt.savefig("../figs/visual_comparison_radiance_O013490.png", dpi=300, bbox_inches='tight')


###############################################################################
# Visualize radiances for image O012791 across different angles
# Save to: "../figs/visual_comparison_radiance_O012791.png"
###############################################################################

grid_df_3 = labeled_image_3.pivot(index='y', columns='x', values='angle_df')
grid_cf_3 = labeled_image_3.pivot(index='y', columns='x', values='angle_cf')
grid_bf_3 = labeled_image_3.pivot(index='y', columns='x', values='angle_bf')
grid_af_3 = labeled_image_3.pivot(index='y', columns='x', values='angle_af')
grid_an_3 = labeled_image_3.pivot(index='y', columns='x', values='angle_an')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of Radiance Across Viewing Angles: Image O012791', fontsize=16)

# Image 3 expert labels
axes[0, 0].imshow(grid_3, cmap=cmap, norm=norm, origin='upper')
axes[0, 0].set_title('Expert Cloud Labels')

# Df angle
im1 = axes[0, 1].imshow(grid_df_3, cmap='plasma_r', origin='upper')
axes[0, 1].set_title('Df Angle (70.5°) Radiance')

# Cf angle
im2 = axes[0, 2].imshow(grid_cf_3, cmap='plasma_r', origin='upper')
axes[0, 2].set_title('Cf Angle (60.0°) Radiance')

# Bf angle
im3 = axes[1, 0].imshow(grid_bf_3, cmap='plasma_r', origin='upper')
axes[1, 0].set_title('Bf Angle (45.6°) Radiance')

# Af angle
im4 = axes[1, 1].imshow(grid_af_3, cmap='plasma_r', origin='upper')
axes[1, 1].set_title('Af Angle (26.1°) Radiance')

# An angle
im5 = axes[1, 2].imshow(grid_an_3, cmap='plasma_r', origin='upper')
axes[1, 2].set_title('An Angle (0.0°) Radiance')

# Remove ticks from all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
fig.colorbar(im5, cax=cbar_ax, orientation='horizontal', label='Radiance')

plt.savefig("../figs/visual_comparison_radiance_O012791.png", dpi=300, bbox_inches='tight')


###############################################################################
# Grouped boxplots of angle radiances by expert label classification
# Save to: "../figs/boxplot_radiance_by_label.png"
###############################################################################

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

# Df angle
sns.boxplot(data=labeled_images, x='cloud', y='angle_df', ax=axes[0, 0], flierprops={'alpha': 0.3, 'markersize': 3})
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Radiance')
axes[0, 0].set_title('Df Angle (70.5°) Radiance by Expert Label')
axes[0, 0].set_xticks([0, 1, 2])
axes[0, 0].set_xticklabels(['Not Cloud', 'Unlabeled', 'Cloud'])
axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)

# Cf angle
sns.boxplot(data=labeled_images, x='cloud', y='angle_cf', ax=axes[0, 1], flierprops={'alpha': 0.3, 'markersize': 3})
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Radiance')
axes[0, 1].set_title('Cf Angle (60.0°) Radiance by Expert Label')
axes[0, 1].set_xticks([0, 1, 2])
axes[0, 1].set_xticklabels(['Not Cloud', 'Unlabeled', 'Cloud'])
axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

# Bf angle
sns.boxplot(data=labeled_images, x='cloud', y='angle_bf', ax=axes[0, 2], flierprops={'alpha': 0.3, 'markersize': 3})
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel('Radiance')
axes[0, 2].set_title('Bf Angle (45.6°) Radiance by Expert Label')
axes[0, 2].set_xticks([0, 1, 2])
axes[0, 2].set_xticklabels(['Not Cloud', 'Unlabeled', 'Cloud'])
axes[0, 2].grid(True, axis='y', linestyle='--', alpha=0.7)

# Af angle
sns.boxplot(data=labeled_images, x='cloud', y='angle_af', ax=axes[1, 0], flierprops={'alpha': 0.3, 'markersize': 3})
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Radiance')
axes[1, 0].set_title('Af Angle (26.1°) Radiance by Expert Label')
axes[1, 0].set_xticks([0, 1, 2])
axes[1, 0].set_xticklabels(['Not Cloud', 'Unlabeled', 'Cloud'])
axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)

# An angle
sns.boxplot(data=labeled_images, x='cloud', y='angle_an', ax=axes[1, 1], flierprops={'alpha': 0.3, 'markersize': 3})
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Radiance')
axes[1, 1].set_title('An Angle (0.0°) Radiance by Expert Label')
axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_xticklabels(['Not Cloud', 'Unlabeled', 'Cloud'])
axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

# Hide empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("../figs/boxplot_radiance_by_label.png", dpi=300, bbox_inches='tight')


###############################################################################
# Boxplot comparing change in radiances with decreasing camera angle
# Save to: "../figs/boxplot_radiance_by_angle.png"
###############################################################################

# Melt angle columns into a single column for plotting
angle_cols = ['angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an']
id_cols = ['y', 'x', 'ndai', 'sd', 'corr', 'cloud', 'image']

labeled_images_long = labeled_images.melt(
    id_vars=id_cols,
    value_vars=angle_cols,
    var_name='angle',
    value_name='radiance'
)

# Partition long df by cloud classification
labeled_images_long_cloud = labeled_images_long[labeled_images_long['cloud'] == 1]
labeled_images_long_unlabeled = labeled_images_long[labeled_images_long['cloud'] == 0]
labeled_images_long_no_cloud = labeled_images_long[labeled_images_long['cloud'] == -1]

# Boxplot comparing change in radiances with decreasing camera angle
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Not cloud
sns.boxplot(data=labeled_images_long_no_cloud, x='angle', y='radiance', ax=axes[0], flierprops={'alpha': 0.3, 'markersize': 3})
axes[0].set_xlabel('Angle')
axes[0].set_ylabel('Radiance')
axes[0].set_title('Not Cloud')
axes[0].set_xticks([0, 1, 2, 3, 4])
axes[0].set_xticklabels(['Df (70.5°)', 'Cf (60.0°)', 'Bf (45.6°)', 'Af (26.1°)', 'An (0.0°)'])
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)

# Unlabeled
sns.boxplot(data=labeled_images_long_unlabeled, x='angle', y='radiance', ax=axes[1], flierprops={'alpha': 0.3, 'markersize': 3})
axes[1].set_xlabel('Angle')
axes[1].set_ylabel('Radiance')
axes[1].set_title('Unlabeled')
axes[1].set_xticks([0, 1, 2, 3, 4])
axes[1].set_xticklabels(['Df (70.5°)', 'Cf (60.0°)', 'Bf (45.6°)', 'Af (26.1°)', 'An (0.0°)'])
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

# Cloud
sns.boxplot(data=labeled_images_long_cloud, x='angle', y='radiance', ax=axes[2], flierprops={'alpha': 0.3, 'markersize': 3})
axes[2].set_xlabel('Angle')
axes[2].set_ylabel('Radiance')
axes[2].set_title('Cloud')
axes[2].set_xticks([0, 1, 2, 3, 4])
axes[2].set_xticklabels(['Df (70.5°)', 'Cf (60.0°)', 'Bf (45.6°)', 'Af (26.1°)', 'An (0.0°)'])
axes[2].grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("../figs/boxplot_radiance_by_angle.png", dpi=300, bbox_inches='tight')


###############################################################################
# Grouped boxplots comparing change in radiances with decreasing camera angle
# Save to: "../figs/boxplot_radiance_filtered.png"
###############################################################################

# Filter out unlabeled, keep only cloud and not cloud
labeled_images_long_filtered = labeled_images_long[labeled_images_long['cloud'] != 0].copy()

# Create a more readable label for the cloud column
labeled_images_long_filtered['label'] = labeled_images_long_filtered['cloud'].map({-1: 'Not Cloud', 1: 'Cloud'})

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=labeled_images_long_filtered, 
    x='angle', 
    y='radiance', 
    hue='label',
    hue_order=['Cloud', 'Not Cloud'],
    ax=ax, 
    flierprops={'alpha': 0.2, 'markersize': 3},
    palette={'Cloud': 'skyblue', 'Not Cloud': 'lightsalmon'},
    gap=0.1
)

ax.set_xlabel('Angle')
ax.set_ylabel('Radiance')
ax.set_title('Boxplot of Radiance by Camera Angle, Grouped by Cloud Classification')
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['Df (70.5°)', 'Cf (60.0°)', 'Bf (45.6°)', 'Af (26.1°)', 'An (0.0°)'])
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Expert Label')

plt.tight_layout()
plt.savefig("../figs/boxplot_radiance_filtered.png", dpi=300, bbox_inches='tight')


###############################################################################
# Boxplot comparing NDAI and expert cloud labels
# Save to: "../figs/boxplot_ndai_filtered.png"
###############################################################################

fig, ax = plt.subplots(figsize=(12, 6))

# Create a more readable label for the cloud column
labeled_images_filtered['label'] = labeled_images_filtered['cloud'].map({-1: 'Not Cloud', 1: 'Cloud'})

sns.boxplot(
    data=labeled_images_filtered,
    x='image',
    y='ndai',
    hue='label',
    hue_order=['Cloud', 'Not Cloud'],
    ax=ax, 
    flierprops={'alpha': 0.2, 'markersize': 3},
    palette={'Cloud': 'skyblue', 'Not Cloud': 'lightsalmon'},
    gap=0.1
)

ax.set_xlabel('Image')
ax.set_ylabel('NDAI')
ax.set_title('Boxplot Comparison of NDAI and Cloud Classification')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Expert Label')

plt.tight_layout()
plt.savefig("../figs/boxplot_ndai_filtered.png", dpi=300, bbox_inches='tight')


###############################################################################
# Boxplot comparing SD and expert cloud labels
# Save to: "../figs/boxplot_sd_filtered.png"
###############################################################################

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=labeled_images_filtered,
    x='image',
    y='sd',
    hue='label',
    hue_order=['Cloud', 'Not Cloud'],
    ax=ax, 
    flierprops={'alpha': 0.2, 'markersize': 3},
    palette={'Cloud': 'skyblue', 'Not Cloud': 'lightsalmon'},
    gap=0.1
)

ax.set_xlabel('Image')
ax.set_ylabel('SD')
ax.set_title('Boxplot Comparison of SD and Cloud Classification')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Expert Label', loc='upper right')

plt.tight_layout()
plt.savefig("../figs/boxplot_sd_filtered.png", dpi=300, bbox_inches='tight')


###############################################################################
# Boxplot comparing CORR and expert cloud labels
# Save to: "../figs/boxplot_corr_filtered.png"
###############################################################################

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=labeled_images_filtered,
    x='image',
    y='corr',
    hue='label',
    hue_order=['Cloud', 'Not Cloud'],
    ax=ax, 
    flierprops={'alpha': 0.2, 'markersize': 3},
    palette={'Cloud': 'skyblue', 'Not Cloud': 'lightsalmon'},
    gap=0.1
)

ax.set_xlabel('Image')
ax.set_ylabel('CORR')
ax.set_title('Boxplot Comparison of CORR and Cloud Classification')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Expert Label', loc='lower left')

plt.tight_layout()
plt.savefig("../figs/boxplot_corr_filtered.png", dpi=300, bbox_inches='tight')


###############################################################################
# Scatterplot of CORR vs NDAI
# Save to: "../figs/scatter_corr_ndai_ndai.png"
###############################################################################

# Scatterplot of CORR vs NDAI for each image, colored by cloud status (excluding unlabeled)
images_filtered = [
    labeled_image_1[labeled_image_1['cloud'] != 0],
    labeled_image_2[labeled_image_2['cloud'] != 0],
    labeled_image_3[labeled_image_3['cloud'] != 0]
]
image_names = ['O013257', 'O013490', 'O012791']

color_map = {-1: 'lightsalmon', 1: 'skyblue'}
label_map = {-1: 'Not Cloud', 1: 'Cloud'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, df, name in zip(axes, images_filtered, image_names):
    for cloud_val in [-1, 1]:
        subset = df[df['cloud'] == cloud_val]
        ax.scatter(
            subset['corr'], subset['ndai'],
            c=color_map[cloud_val],
            label=label_map[cloud_val],
            alpha=0.2,
            s=2
        )
    ax.set_xlabel('CORR')
    ax.set_ylabel('NDAI')
    ax.set_title(f'CORR vs NDAI: {name}')
    ax.grid(True, linestyle='--', alpha=0.5)

legend = axes[-1].legend(title='Cloud Status', markerscale=3, loc='upper right')
for lh in legend.legend_handles:
    lh.set_alpha(1)

plt.tight_layout()
plt.savefig("../figs/scatter_corr_ndai.png", dpi=300, bbox_inches='tight')


###############################################################################
# Scatterplot of SD vs NDAI
# Save to: "../figs/scatter_sd_ndai.png"
###############################################################################

# Scatterplot of SD vs NDAI for each image, colored by cloud status (excluding unlabeled)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, df, name in zip(axes, images_filtered, image_names):
    for cloud_val in [-1, 1]:
        subset = df[df['cloud'] == cloud_val]
        ax.scatter(
            subset['sd'], subset['ndai'],
            c=color_map[cloud_val],
            label=label_map[cloud_val],
            alpha=0.2,
            s=2
        )
    ax.set_xlabel('SD')
    ax.set_ylabel('NDAI')
    ax.set_title(f'SD vs NDAI: {name}')
    ax.grid(True, linestyle='--', alpha=0.5)

legend = axes[-1].legend(title='Cloud Status', markerscale=3, loc='upper right')
for lh in legend.legend_handles:
    lh.set_alpha(1)

plt.tight_layout()
plt.savefig("../figs/scatter_sd_ndai.png", dpi=300, bbox_inches='tight')


###############################################################################
# Scatterplot of SD VS CORR
# Save to: "../figs/scatter_sd_corr.png"
###############################################################################

# Scatterplot of CORR vs SD for each image, colored by cloud status (excluding unlabeled)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, df, name in zip(axes, images_filtered, image_names):
    for cloud_val in [-1, 1]:
        subset = df[df['cloud'] == cloud_val]
        ax.scatter(
            subset['corr'], subset['sd'],
            c=color_map[cloud_val],
            label=label_map[cloud_val],
            alpha=0.2,
            s=2
        )
    ax.set_xlabel('CORR')
    ax.set_ylabel('SD')
    ax.set_title(f'CORR vs SD: {name}')
    ax.grid(True, linestyle='--', alpha=0.5)

legend = axes[-1].legend(title='Cloud Status', markerscale=3, loc='upper right')
for lh in legend.legend_handles:
    lh.set_alpha(1)

plt.tight_layout()
plt.savefig("../figs/scatter_corr_sd.png", dpi=300, bbox_inches='tight')


###############################################################################
# Visual comparison between NDAI and expert labels
# Save to: "../figs/visual_comparison_ndai.png"
###############################################################################

grid_ndai_1 = labeled_image_1.pivot(index='y', columns='x', values='ndai')
grid_ndai_2 = labeled_image_2.pivot(index='y', columns='x', values='ndai')
grid_ndai_3 = labeled_image_3.pivot(index='y', columns='x', values='ndai')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of NDAI and Expert Cloud Labels', fontsize=16)

# Image 1 expert labels
axes[0, 0].imshow(grid_1, cmap=cmap, norm=norm, origin='upper')
axes[0, 0].set_title('O013257 Expert Cloud Labels')

# Image 2 expert labels
axes[0, 1].imshow(grid_2, cmap=cmap, norm=norm, origin='upper')
axes[0, 1].set_title('O013490 Expert Cloud Labels')

# Image 3 expert labels
axes[0, 2].imshow(grid_3, cmap=cmap, norm=norm, origin='upper')
axes[0, 2].set_title('O012791 Expert Cloud Labels')

# Image 1 NDAI
axes[1, 0].imshow(grid_ndai_1, cmap='RdBu_r', origin='upper')
axes[1, 0].set_title('O013257 NDAI')

# Image 2 NDAI
axes[1, 1].imshow(grid_ndai_2, cmap='RdBu_r', origin='upper')
axes[1, 1].set_title('O013490 NDAI')

# Image 3 NDAI
im5 = axes[1, 2].imshow(grid_ndai_3, cmap='RdBu_r', origin='upper')
axes[1, 2].set_title('O012791 NDAI')

# Remove ticks from all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
cbar = fig.colorbar(im5, cax=cbar_ax, orientation='horizontal', label='NDAI')
cbar.ax.xaxis.set_major_locator(MultipleLocator(0.1)) 

plt.savefig("../figs/visual_comparison_ndai.png", dpi=300, bbox_inches='tight')


###############################################################################
# Visual comparison between SD and expert labels
# Save to: "../figs/visual_comparison_sd.png"
###############################################################################

grid_sd_1 = labeled_image_1.pivot(index='y', columns='x', values='sd')
grid_sd_2 = labeled_image_2.pivot(index='y', columns='x', values='sd')
grid_sd_3 = labeled_image_3.pivot(index='y', columns='x', values='sd')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of SD and Expert Cloud Labels', fontsize=16)

# Image 1 expert labels
axes[0, 0].imshow(grid_1, cmap=cmap, norm=norm, origin='upper')
axes[0, 0].set_title('O013257 Expert Cloud Labels')

# Image 2 expert labels
axes[0, 1].imshow(grid_2, cmap=cmap, norm=norm, origin='upper')
axes[0, 1].set_title('O013490 Expert Cloud Labels')

# Image 3 expert labels
axes[0, 2].imshow(grid_3, cmap=cmap, norm=norm, origin='upper')
axes[0, 2].set_title('O012791 Expert Cloud Labels')

# Image 1 SD
axes[1, 0].imshow(grid_sd_1, cmap='viridis', origin='upper', vmin=0, vmax=9000)
axes[1, 0].set_title('O013257 SD')

# Image 2 SD
axes[1, 1].imshow(grid_sd_2, cmap='viridis', origin='upper', vmin=0, vmax=9000)
axes[1, 1].set_title('O013490 SD')

# Image 3 sd
im5 = axes[1, 2].imshow(grid_sd_3, cmap='viridis', origin='upper', vmin=0, vmax=9000)
axes[1, 2].set_title('O012791 SD')

# Remove ticks from all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
cbar = fig.colorbar(im5, cax=cbar_ax, orientation='horizontal', label='SD')
cbar.ax.xaxis.set_major_locator(MultipleLocator(1000)) 

plt.savefig("../figs/visual_comparison_sd.png", dpi=300, bbox_inches='tight')


###############################################################################
# Visual comparison between CORR and expert labels
# Save to: "../figs/visual_comparison_corr.png"
###############################################################################

grid_corr_1 = labeled_image_1.pivot(index='y', columns='x', values='corr')
grid_corr_2 = labeled_image_2.pivot(index='y', columns='x', values='corr')
grid_corr_3 = labeled_image_3.pivot(index='y', columns='x', values='corr')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of CORR and Expert Cloud Labels', fontsize=16)

# Image 1 expert labels
axes[0, 0].imshow(grid_1, cmap=cmap, norm=norm, origin='upper')
axes[0, 0].set_title('O013257 Expert Cloud Labels')

# Image 2 expert labels
axes[0, 1].imshow(grid_2, cmap=cmap, norm=norm, origin='upper')
axes[0, 1].set_title('O013490 Expert Cloud Labels')

# Image 3 expert labels
axes[0, 2].imshow(grid_3, cmap=cmap, norm=norm, origin='upper')
axes[0, 2].set_title('O012791 Expert Cloud Labels')

# Image 1 CORR
axes[1, 0].imshow(grid_corr_1, cmap='RdBu_r', origin='upper', vmin=-1, vmax=1)
axes[1, 0].set_title('O013257 CORR')

# Image 2 CORR
axes[1, 1].imshow(grid_corr_2, cmap='RdBu_r', origin='upper', vmin=-1, vmax=1)
axes[1, 1].set_title('O013490 CORR')

# Image 3 CORR
im5 = axes[1, 2].imshow(grid_corr_3, cmap='RdBu_r', origin='upper', vmin=-1, vmax=1)
axes[1, 2].set_title('O012791 CORR')

# Remove ticks from all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.21, 0.02, 0.6, 0.02])
cbar = fig.colorbar(im5, cax=cbar_ax, orientation='horizontal', label='CORR')
cbar.ax.xaxis.set_major_locator(MultipleLocator(0.1)) 

plt.savefig("../figs/visual_comparison_corr.png", dpi=300, bbox_inches='tight')