import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, permutation_importance
import matplotlib.pyplot as plt


def create_blocks(df, image_name, grid_size=9):
    """Assign block ids to rows using xy coords for cross validation."""
    df = df.copy()

    df['x_bin'] = pd.qcut(df['x'], q=grid_size, labels=False, duplicates='drop')
    df['y_bin'] = pd.qcut(df['y'], q=grid_size, labels=False, duplicates='drop')

    df['block_id'] = image_name + '_' + df['x_bin'].astype(str) + '_' + df['y_bin'].astype(str)
    return df


def load_and_split_data(data_dir):
    """Perform train/test split and return arrays for modeling."""
    # Load the datasets
    dfs = []
    cols_to_drop = ['y', 'x', 'ndai', 'sd', 'corr', 'angle_df', 'angle_cf', 
                   'angle_bf', 'angle_af', 'angle_an', 'cloud', 'image']
    
    for image_name in ['O013257', 'O013490', 'O012791']:

        df_embedding = pd.read_csv(f'{data_dir}/{image_name}_with_embeddings.csv')

        df_enriched = pd.read_csv(f'{data_dir}/{image_name}_enriched.csv')
        df_enriched.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Combine the new features and autoencoder features
        df = pd.concat([df_embedding, df_enriched], axis=1)

        # Create blocks for training set
        if image_name in ['O013257', 'O012791']:
            df = create_blocks(df, image_name)
        dfs.append(df)

    # O013257 and O012791 are used for training, while O013490 is used for testing
    df_train = pd.concat([dfs[0], dfs[2]], ignore_index=True)
    df_test = dfs[1].copy()

    # Only keep labeled pixels
    df_train = df_train[df_train['cloud'] != 0].copy()
    df_test = df_test[df_test['cloud'] != 0].copy()

    # Prepare features and labels
    features = ['ndai', 'sd', 'corr', 'angle_df', 'angle_cf', 
                'angle_bf', 'angle_af', 'angle_an', 
                'ae0', 'ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 
                'ae8', 'ae9', 'ae10', 'ae11', 'ae12', 'ae13', 'ae14', 'ae15', 
                'rad_ratio', 'rad_range', 'rad_std', 'ndai_local_mean', 'ndai_local_std', 
                'sd_local_mean', 'sd_local_std', 'ndai_gradient']

    X_train = df_train[features].values
    y_train = df_train['cloud'].values
    blocks = df_train['block_id'].values

    X_test = df_test[features].values
    y_test = df_test['cloud'].values

    print(f'Train size: {len(X_train)}')
    print(f'Test size: {len(X_test)}')
    print(f'Number of blocks: {len(np.unique(blocks))}')

    return X_train, y_train, X_test, y_test, features, blocks, df_test


# Cross-Validation based on block_id
def evaluate_with_cv(X_train, y_train, blocks, models):
    """Compute 5-fold ROC-AUC for each model."""
    rows = []
    for name, model in models.items():
        cv_auc = cross_val_score(
            model,
            X_train,
            y_train, 
            groups=blocks,
            cv=GroupKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        rows.append({
            'Model': name,
            'CV ROC-AUC Mean': cv_auc.mean(),
            'CV ROC-AUC Std': cv_auc.std()
        })
        print(f'{name} Mean CV AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})')

    cv_table = pd.DataFrame(rows).sort_values('CV ROC-AUC Mean', ascending=False).reset_index(drop=True)
    cv_table.insert(0, 'Rank', np.arange(1, len(cv_table) + 1))

    return cv_table

def tune_tree_models(X_train, y_train, blocks):
    """Perform grid search CV to find best params for random forest and gradient boosted trees."""
    cv = GroupKFold(n_splits=5)

    # Define parameter grids
    search_space = {
        'Random Forest': (
            RandomForestClassifier(random_state=214, n_jobs=1),
            {
                'n_estimators': [100, 300, 500],
                'max_depth': [20, 40, None],
                'min_samples_leaf': [1, 5, 20],
            },
        ),
        'Gradient Boosting': (
            HistGradientBoostingClassifier(random_state=214, early_stopping=True),
            {
                'max_iter': [100, 300, 500],
                'max_depth': [5, 10, None],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_leaf': [10, 20, 50],
            },
        ),
    }

    best_models = {}
    for name, (estimator, param_grid) in search_space.items():
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            refit=True
        )

        gs.fit(X_train, y_train, groups=blocks)
        best_models[name] = gs.best_estimator_
        print(f'{name} best params: {gs.best_params_}')
        print(f'{name} best CV AUC: {gs.best_score_}')

    return best_models

def evaluate_on_test_set(X_train, y_train, X_test, y_test, models):
    """Fit models, compute test ROC-AUC, save grouped ROC plot, and return the best model."""
    best_name = None
    best_auc = 0
    best_model = None

    plt.figure()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        test_auc = auc(fpr, tpr)
        print(f'{name} Test AUC: {test_auc:.4f}')
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {test_auc:.4f})')

        # Update best model
        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name
            best_model = model
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of all models')
    plt.legend(loc='lower right')
    plt.savefig('../figs/ROC_all_models.png')
    plt.close()

    print(f'Best model is {best_name} with AUC = {best_auc:.4f}')

    return best_model

def plot_convergence(best_model, X_train, y_train, X_test, y_test):
    """Plot convergence diagnostics of Gradient Boosting."""
    train_aucs = []
    val_aucs = []
    iters = range(50, best_model.max_iter + 1, 50)

    for n in iters:
        m = HistGradientBoostingClassifier(
            max_iter=n,
            max_depth=best_model.max_depth,
            learning_rate=best_model.learning_rate,
            min_samples_leaf=best_model.min_samples_leaf,
            random_state=214, early_stopping=False
        )
        m.fit(X_train, y_train)
        train_aucs.append(roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]))
        val_aucs.append(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))

    plt.plot(iters, train_aucs, label='Train AUC')
    plt.plot(iters, val_aucs, label='Test AUC')
    plt.xlabel('Number of boosting iterations')
    plt.ylabel('AUC')
    plt.title('Convergence of Gradient Boosting Classifier')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('../figs/convergence.png')
    plt.close()

def analyze_feature_importance(model, X_train, y_train, X_test, y_test, features):
    "Find permutation importance on test set."
    model.fit(X_train, y_train)

    perm_result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=214,
        n_jobs=-1
    )
    
    df_result = pd.DataFrame({
        "feature": features,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    return df_result

def posthoc_error_analysis(best_model, X_test, y_test, features, df_test):
    "Perform post-hoc error analysis."

    y_pred = (best_model.predict_proba(X_test)[:, 1]>= 0.5).astype(int)
    y_true = (y_test == 1).astype(int)

    correct = (y_pred == y_true)
    # False positives (Predict not cloud as cloud)
    fp = (y_pred == 1) & (y_true == 0)  
    # False negatives (Predict cloud as not cloud)
    fn = (y_pred == 0) & (y_true == 1)  

    print(f'False positives: {fp.mean()*100:.2f}%')
    print(f'False negatives: {fn.mean()*100:.2f}%')

    x_coords = df_test['x'].values
    y_coords = df_test['y'].values

    # 1. Plot spatial distribution of errors
    plt.scatter(x_coords[correct],y_coords[correct], s=1, alpha=0.7, color='lightgrey', label='Correct')
    plt.scatter(x_coords[fp], y_coords[fp], s=1, label='FP')
    plt.scatter(x_coords[fn], y_coords[fn], s=1, label='FN')
    # Invert y-axis to match image coordinate system
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Spatial Distribution of Errors')
    plt.legend(markerscale=6, loc='lower right')
    plt.tight_layout()
    plt.savefig('../figs/error_spatial.png')
    plt.close()

    # 2. Plot feature distributions of errors
    selected_features = ['ndai', 'sd', 'sd_local_mean']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(len(selected_features)):       
        f = selected_features[i]
        idx = features.index(f)
        ax = axes[i]

        ax.hist(X_test[correct, idx], bins=100, density=True, color='green', alpha=0.3, label='Correct')
        ax.hist(X_test[fp, idx], bins=100, density=True, alpha=0.6, label='FP')
        ax.hist(X_test[fn, idx], bins=100, density=True, alpha=0.6, label='FN')

        ax.set_xlabel(f)
        ax.set_ylabel('Density')
        ax.legend()

    fig.suptitle('Error distributions in important features', fontsize=14)

    plt.tight_layout()
    plt.savefig('../figs/error_feature_distributions.png')
    plt.close()




if __name__ == "__main__":
    # Use logistic regression as baseline model
    baseline_model = {
        'Logistic Regression': LogisticRegression(
            max_iter=10000, solver='saga', tol=1e-3, random_state=214
        )
    }

    # Create training ang testing splits
    X_train, y_train, X_test, y_test, features, blocks, df_test = load_and_split_data('../data/image_data')
    print('\n')

    # Perform hyperparameter tuning on random forest and gradient boosting models
    # Save best parameters to models
    tuned_tree_models = tune_tree_models(X_train, y_train, blocks)
    models = {**baseline_model, **tuned_tree_models}

    cv_table = evaluate_with_cv(X_train, y_train, blocks, models)
    print(cv_table.to_string(index=False))
    print('\n')

    best_model = evaluate_on_test_set(X_train, y_train, X_test, y_test, models)

    plot_convergence(best_model, X_train, y_train, X_test, y_test)

    print(analyze_feature_importance(best_model, X_train, y_train, X_test, y_test, features))

    posthoc_error_analysis(best_model, X_test, y_test, features, df_test)