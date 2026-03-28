import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from predictive_modeling import load_and_split_data

features = ['ndai', 'sd', 'corr', 'angle_df', 'angle_cf',
            'angle_bf', 'angle_af', 'angle_an',
            'ae0', 'ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7',
            'ae8', 'ae9', 'ae10', 'ae11', 'ae12', 'ae13', 'ae14', 'ae15',
            'rad_ratio', 'rad_range', 'rad_std', 'ndai_local_mean',
            'ndai_local_std', 'sd_local_mean', 'sd_local_std', 'ndai_gradient']

X_train, y_train, X_test, y_test, _, blocks = load_and_split_data('../data/image_data')

# also load test image with coords for the spatial plot
df_emb = pd.read_csv('../data/image_data/O013490_with_embeddings.csv')
df_enr = pd.read_csv('../data/image_data/O013490_enriched.csv')
df_enr.drop(columns=['y', 'x', 'ndai', 'sd', 'corr', 'angle_df', 'angle_cf',
                     'angle_bf', 'angle_af', 'angle_an', 'cloud', 'image'],
            inplace=True, errors='ignore')
df_test = pd.concat([df_emb, df_enr], axis=1)
df_test = df_test[df_test['cloud'] != 0].copy()

model = HistGradientBoostingClassifier(max_iter=500, max_depth=10, learning_rate=0.1,
                                       min_samples_leaf=50, random_state=214, early_stopping=True)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
y_test_bin = (y_test == 1).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
ConfusionMatrixDisplay(cm, display_labels=['Not Cloud', 'Cloud']).plot(ax=axes[0, 0], colorbar=False)
axes[0, 0].set_title('Confusion Matrix (threshold = 0.5)')

# precision-recall
prec, rec, _ = precision_recall_curve(y_test_bin, y_prob, pos_label=1)
ap = average_precision_score(y_test_bin, y_prob, pos_label=1)
axes[0, 1].plot(rec, prec, lw=2, label=f'AP = {ap:.4f}')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend()

# calibration
frac_pos, mean_pred = calibration_curve(y_test_bin, y_prob, n_bins=10, pos_label=1)
axes[1, 0].plot(mean_pred, frac_pos, 's-', label='Gradient Boosting')
axes[1, 0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
axes[1, 0].set_xlabel('Mean Predicted Probability')
axes[1, 0].set_ylabel('Fraction Positive')
axes[1, 0].set_title('Calibration Curve')
axes[1, 0].legend()

# permutation importance - shuffle each feature and measure AUC drop
result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                random_state=214, scoring='roc_auc', n_jobs=-1)
idx = np.argsort(result.importances_mean)[-15:]
axes[1, 1].barh(np.array(features)[idx], result.importances_mean[idx],
                xerr=result.importances_std[idx])
axes[1, 1].set_xlabel('Mean AUC decrease')
axes[1, 1].set_title('Permutation Feature Importance (top 15)')

plt.tight_layout()
plt.savefig('../figs/diagnostics_summary.png', dpi=150)
plt.close()
print('Saved diagnostics_summary.png')

# spatial error and probability maps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

error = (y_pred != df_test['cloud'].values).astype(int)
sc0 = axes[0].scatter(df_test['x'], df_test['y'], c=error, cmap='RdYlGn_r', s=0.2, vmin=0, vmax=1)
plt.colorbar(sc0, ax=axes[0], label='Error (1=wrong)')
axes[0].set_title('Spatial Error Map (O013490 test set)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].invert_yaxis()

sc1 = axes[1].scatter(df_test['x'], df_test['y'], c=y_prob, cmap='coolwarm', s=0.2, vmin=0, vmax=1)
plt.colorbar(sc1, ax=axes[1], label='P(cloud)')
axes[1].set_title('Predicted Cloud Probability (O013490 test set)')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('../figs/diagnostics_spatial.png', dpi=150)
plt.close()
print('Saved diagnostics_spatial.png')
