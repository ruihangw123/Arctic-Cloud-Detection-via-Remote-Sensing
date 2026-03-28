import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from predictive_modeling import create_blocks

images = ['O013257', 'O013490', 'O012791']

all_features = [
    'ndai', 'sd', 'corr', 'angle_df', 'angle_cf',
    'angle_bf', 'angle_af', 'angle_an',
    'ae0', 'ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7',
    'ae8', 'ae9', 'ae10', 'ae11', 'ae12', 'ae13', 'ae14', 'ae15',
    'rad_ratio', 'rad_range', 'rad_std', 'ndai_local_mean',
    'ndai_local_std', 'sd_local_mean', 'sd_local_std', 'ndai_gradient',
]
orig_features = all_features[:8]
engi_features = all_features[:8] + all_features[24:]

cols_to_drop = ['y', 'x', 'ndai', 'sd', 'corr', 'angle_df', 'angle_cf',
                'angle_bf', 'angle_af', 'angle_an', 'cloud', 'image']

def load_image(name):
    df_emb = pd.read_csv(f'../data/image_data/{name}_with_embeddings.csv')
    df_enr = pd.read_csv(f'../data/image_data/{name}_enriched.csv')
    df_enr.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return pd.concat([df_emb, df_enr], axis=1)

def make_split(test_image):
    test_df = load_image(test_image)
    train_dfs = []
    for name in [n for n in images if n != test_image]:
        df = create_blocks(load_image(name), name)
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = train_df[train_df['cloud'] != 0]
    test_df = test_df[test_df['cloud'] != 0]
    return train_df, train_df['cloud'].values, test_df, test_df['cloud'].values

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, solver='saga', tol=1e-3, random_state=214),
    'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=1, random_state=214, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingClassifier(max_iter=500, max_depth=10, learning_rate=0.1, min_samples_leaf=50, random_state=214, early_stopping=True),
}
model_colors = {'Logistic Regression': 'tab:blue', 'Random Forest': 'tab:orange', 'Gradient Boosting': 'tab:green'}

# experiment 1: does model ranking hold across all 3 LOO image splits?
print("=== Experiment 1: model ranking across LOO splits ===")
rows1 = []
for test_img in images:
    train_df, y_train, test_df, y_test = make_split(test_img)
    for name, mdl in models.items():
        mdl.fit(train_df[all_features].values, y_train)
        auc = roc_auc_score(y_test, mdl.predict_proba(test_df[all_features].values)[:, 1])
        rows1.append({'Test image': test_img, 'Model': name, 'AUC': auc})
        print(f"  {test_img} / {name}: {auc:.4f}")

df1 = pd.DataFrame(rows1)
print(df1.pivot(index='Test image', columns='Model', values='AUC').round(4))

# experiment 2: does model ranking hold across feature sets?
print("\n=== Experiment 2: model ranking across feature sets (test = O013490) ===")
train_df, y_train, test_df, y_test = make_split('O013490')
feat_sets = {'Original (8)': orig_features, 'Original + Engineered (16)': engi_features, 'All + AE (32)': all_features}
rows2 = []
for feat_label, feats in feat_sets.items():
    for name, mdl in models.items():
        mdl.fit(train_df[feats].values, y_train)
        auc = roc_auc_score(y_test, mdl.predict_proba(test_df[feats].values)[:, 1])
        rows2.append({'Feature set': feat_label, 'Model': name, 'AUC': auc})
        print(f"  {feat_label} / {name}: {auc:.4f}")

df2 = pd.DataFrame(rows2)
print(df2.pivot(index='Feature set', columns='Model', values='AUC').round(4))

# experiment 3: learning curve - does performance hold with less training data?
print("\n=== Experiment 3: learning curve (test = O013490) ===")
train_df, y_train, test_df, y_test = make_split('O013490')
X_tr = train_df[all_features].values
X_te = test_df[all_features].values
rng = np.random.RandomState(214)
rows3 = []
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    n = int(len(X_tr) * frac)
    idx = rng.choice(len(X_tr), n, replace=False)
    mdl = HistGradientBoostingClassifier(max_iter=500, max_depth=10, learning_rate=0.1,
                                         min_samples_leaf=50, random_state=214, early_stopping=True)
    mdl.fit(X_tr[idx], y_train[idx])
    auc = roc_auc_score(y_test, mdl.predict_proba(X_te)[:, 1])
    rows3.append({'frac': frac, 'n': n, 'AUC': auc})
    print(f"  {frac:.0%}  n={n}  AUC={auc:.4f}")

df3 = pd.DataFrame(rows3)

# plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
width = 0.25

pivot1 = df1.pivot(index='Test image', columns='Model', values='AUC')
x = np.arange(len(pivot1))
for i, col in enumerate(pivot1.columns):
    axes[0].bar(x + i * width, pivot1[col], width, label=col, color=model_colors[col], alpha=0.85)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(pivot1.index, rotation=15)
axes[0].set_ylabel('Test AUC')
axes[0].set_ylim(0.6, 1.0)
axes[0].set_title('Model ranking across LOO image splits\n(stability of model selection)')
axes[0].legend(fontsize=8)

pivot2 = df2.pivot(index='Feature set', columns='Model', values='AUC')
pivot2 = pivot2.reindex(['Original (8)', 'Original + Engineered (16)', 'All + AE (32)'])
x2 = np.arange(len(pivot2))
for i, col in enumerate(pivot2.columns):
    axes[1].bar(x2 + i * width, pivot2[col], width, label=col, color=model_colors[col], alpha=0.85)
axes[1].set_xticks(x2 + width)
axes[1].set_xticklabels(['Original\n(8)', '+Engineered\n(16)', '+AE\n(32)'], fontsize=8)
axes[1].set_ylim(0.6, 1.0)
axes[1].set_ylabel('Test AUC')
axes[1].set_title('Model ranking across feature sets\n(stability of feature set decision)')
axes[1].legend(fontsize=8)

axes[2].plot(df3['n'], df3['AUC'], 'o-', color='tab:green', lw=2)
axes[2].set_xlabel('Training set size')
axes[2].set_ylabel('Test AUC')
axes[2].set_title('Learning curve\n(stability w.r.t. training data volume)')
axes[2].set_ylim(0.88, 1.0)
for _, row in df3.iterrows():
    axes[2].annotate(f"{row['AUC']:.4f}", (row['n'], row['AUC']),
                     textcoords='offset points', xytext=(0, 6), fontsize=7, ha='center')

plt.tight_layout()
plt.savefig('../figs/stability.png', dpi=150)
plt.close()
print('\nSaved stability.png')
