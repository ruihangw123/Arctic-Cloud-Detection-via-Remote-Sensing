import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from predictive_modeling import load_and_split_data

features = ['ndai', 'sd', 'corr', 'angle_df', 'angle_cf',
            'angle_bf', 'angle_af', 'angle_an',
            'ae0', 'ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7',
            'ae8', 'ae9', 'ae10', 'ae11', 'ae12', 'ae13', 'ae14', 'ae15',
            'rad_ratio', 'rad_range', 'rad_std', 'ndai_local_mean',
            'ndai_local_std', 'sd_local_mean', 'sd_local_std', 'ndai_gradient']

X_train, y_train, X_test, y_test, _, blocks = load_and_split_data('../data/image_data')

col_names = ['y', 'x', 'ndai', 'sd', 'corr',
             'angle_df', 'angle_cf', 'angle_bf', 'angle_af', 'angle_an', 'cloud']
raw = np.load('../data/image_data/O013490.npz')['arr_0']
df_raw = pd.DataFrame(raw, columns=col_names)
df_raw = df_raw[df_raw['cloud'] != 0].reset_index(drop=True)

model = HistGradientBoostingClassifier(max_iter=500, max_depth=10, learning_rate=0.1,
                                       min_samples_leaf=50, random_state=214, early_stopping=True)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

outcome = np.where(
    (y_pred == 1) & (y_test == 1), 'TP',
    np.where((y_pred == -1) & (y_test == -1), 'TN',
    np.where((y_pred == 1) & (y_test == -1), 'FP', 'FN'))
)
df_raw['outcome'] = outcome
df_raw['y_prob'] = y_prob

# boundary pixels: any 4-neighbor has a different label
# unlabeled (0) neighbors count since experts leave transition zones unlabeled
label_grid = np.zeros((int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1))
label_grid[raw[:, 0].astype(int), raw[:, 1].astype(int)] = raw[:, 10]

def is_boundary(ys, xs, grid):
    lbl = grid[ys, xs]
    neighbors = [
        grid[np.clip(ys-1, 0, grid.shape[0]-1), xs],
        grid[np.clip(ys+1, 0, grid.shape[0]-1), xs],
        grid[ys, np.clip(xs-1, 0, grid.shape[1]-1)],
        grid[ys, np.clip(xs+1, 0, grid.shape[1]-1)],
    ]
    return np.any([n != lbl for n in neighbors], axis=0)

df_raw['is_boundary'] = is_boundary(df_raw['y'].values.astype(int),
                                    df_raw['x'].values.astype(int), label_grid)

print("Fraction of boundary pixels by outcome:")
print(df_raw.groupby('outcome')['is_boundary'].mean().round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# predicted probability distributions
for label, color in [('FP', 'tab:orange'), ('FN', 'tab:red'), ('TP', 'tab:blue'), ('TN', 'tab:green')]:
    sub = df_raw.loc[df_raw['outcome'] == label, 'y_prob']
    axes[0].hist(sub, bins=40, alpha=0.6, color=color, label=f'{label} (n={len(sub)})', density=True)
axes[0].set_xlabel('Predicted P(cloud)')
axes[0].set_ylabel('Density')
axes[0].set_title('Predicted probability by outcome')
axes[0].legend(fontsize=8)

# boundary rate by outcome
boundary_rate = df_raw.groupby('outcome')['is_boundary'].mean()
colors = {'TP': 'tab:blue', 'TN': 'tab:green', 'FP': 'tab:orange', 'FN': 'tab:red'}
axes[1].bar(boundary_rate.index, boundary_rate.values,
            color=[colors[o] for o in boundary_rate.index])
axes[1].set_ylabel('Fraction that are boundary pixels')
axes[1].set_title('Boundary pixel rate by outcome')
axes[1].set_ylim(0, 1)
for i, (o, v) in enumerate(boundary_rate.items()):
    axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

# feature distributions by outcome
top_feats = ['sd_local_mean', 'ndai', 'ndai_local_std']
outcome_order = ['TN', 'FP', 'TP', 'FN']
x_positions = np.arange(len(outcome_order))
width = 0.22
for i, feat in enumerate(top_feats):
    feat_idx = features.index(feat)
    data = [X_test[df_raw['outcome'] == o, feat_idx] for o in outcome_order]
    bp = axes[2].boxplot(data, positions=x_positions + (i - 1) * width,
                         widths=width * 0.85, patch_artist=True,
                         medianprops=dict(color='black', linewidth=1.5),
                         showfliers=False)
    for patch in bp['boxes']:
        patch.set_alpha(0.6)
axes[2].set_xticks(x_positions)
axes[2].set_xticklabels(outcome_order)
axes[2].set_ylabel('Feature value (raw scale)')
axes[2].set_title('Feature distributions by outcome\n(sd_local_mean, ndai, ndai_local_std)')
handles = [plt.Rectangle((0,0),1,1, fc='white', ec='black', label=f) for f in top_feats]
axes[2].legend(handles=handles, fontsize=7)

plt.tight_layout()
plt.savefig('../figs/posthoc_eda.png', dpi=150)
plt.close()
print('Saved posthoc_eda.png')

print("\nMedian predicted probability by outcome:")
print(df_raw.groupby('outcome')['y_prob'].median().round(3))

print("\nMean sd_local_mean by outcome:")
feat_idx = features.index('sd_local_mean')
for o in ['FP', 'FN', 'TP', 'TN']:
    print(f"  {o}: {X_test[df_raw['outcome'] == o, feat_idx].mean():.3f}")
