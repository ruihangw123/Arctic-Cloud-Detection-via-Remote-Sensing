"""PCS stability analysis with bootstrap, preprocessing, hyperparameter, and noise perturbations."""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def _save_fig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _score(clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_te, prob),
        "f1":      f1_score(y_te, pred, zero_division=0),
    }


def _base_clf(name, seed=42):
    if name == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "gbm":
        return GradientBoostingClassifier(n_estimators=100, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def bootstrap_stability(X_train, y_train, X_test, y_test,
                        model_name="gbm", n_bootstrap=50, seed=42, output_dir=None):
    """Resample training data and measure metric variability across resamples."""
    print(f"\n[Stability] Bootstrap ({n_bootstrap} iterations, model={model_name})")
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        scores = _score(_base_clf(model_name, seed=int(rng.integers(1e6))),
                        X_train[idx], y_train[idx], X_test, y_test)
        aucs.append(scores["roc_auc"])
        f1s.append(scores["f1"])

    aucs, f1s = np.array(aucs), np.array(f1s)
    print(f"  ROC-AUC: {aucs.mean():.4f} ± {aucs.std():.4f}  "
          f"[{aucs.min():.4f}, {aucs.max():.4f}]")
    print(f"  F1:      {f1s.mean():.4f} ± {f1s.std():.4f}  "
          f"[{f1s.min():.4f}, {f1s.max():.4f}]")

    if output_dir:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, vals, label, color in [
            (axes[0], aucs, "ROC-AUC", "#1f77b4"),
            (axes[1], f1s,  "F1",      "#ff7f0e"),
        ]:
            ax.hist(vals, bins=20, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(vals.mean(), color="black", linestyle="--", label=f"mean={vals.mean():.4f}")
            ax.axvline(vals.mean() - vals.std(), color="red", linestyle=":", linewidth=0.8)
            ax.axvline(vals.mean() + vals.std(), color="red", linestyle=":", linewidth=0.8,
                       label=f"±1σ={vals.std():.4f}")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        fig.suptitle(f"Bootstrap Stability — {model_name} (n={n_bootstrap})", fontsize=12)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"stability_bootstrap_{model_name}.png")

    return {"auc_mean": aucs.mean(), "auc_std": aucs.std(),
            "f1_mean":  f1s.mean(),  "f1_std":  f1s.std()}


def preprocessing_stability(X_train_raw, y_train, X_test_raw, y_test,
                             model_name="gbm", seed=42, output_dir=None):
    """Test sensitivity to different winsorization thresholds."""
    print(f"\n[Stability] Preprocessing perturbation (model={model_name})")
    configs = [
        ("no winsor",   0.00, 1.00),
        ("1%–99%",      0.01, 0.99),
        ("2%–98%",      0.02, 0.98),
        ("5%–95%",      0.05, 0.95),
        ("10%–90%",     0.10, 0.90),
    ]
    results = []
    for label, q_lo, q_hi in configs:
        lo = np.quantile(X_train_raw, q_lo, axis=0)
        hi = np.quantile(X_train_raw, q_hi, axis=0)
        X_tr = np.clip(X_train_raw, lo, hi)
        X_te = np.clip(X_test_raw,  lo, hi)
        scores = _score(_base_clf(model_name, seed), X_tr, y_train, X_te, y_test)
        results.append((label, scores["roc_auc"], scores["f1"]))
        print(f"  {label:12s}  AUC={scores['roc_auc']:.4f}  F1={scores['f1']:.4f}")

    if output_dir:
        labels = [r[0] for r in results]
        aucs   = [r[1] for r in results]
        f1s    = [r[2] for r in results]
        x = np.arange(len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, vals, ylabel, color in [
            (axes[0], aucs, "ROC-AUC", "#1f77b4"),
            (axes[1], f1s,  "F1",      "#ff7f0e"),
        ]:
            ax.bar(x, vals, color=color, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ymin = min(vals) - 0.02
            ax.set_ylim(ymin, 1.0)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Preprocessing Stability — {model_name}", fontsize=12)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"stability_preprocessing_{model_name}.png")

    return results


def hyperparameter_stability(X_train, y_train, X_test, y_test,
                              model_name="rf", output_dir=None):
    """Compare hyperparameter configs within the same model family."""
    print(f"\n[Stability] Hyperparameter perturbation (model={model_name})")

    if model_name == "rf":
        configs = [
            ("n=50,  d=None", RandomForestClassifier(n_estimators=50,  max_depth=None, random_state=42, n_jobs=-1)),
            ("n=100, d=None", RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
            ("n=200, d=None", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)),
            ("n=100, d=10",   RandomForestClassifier(n_estimators=100, max_depth=10,   random_state=42, n_jobs=-1)),
            ("n=100, d=20",   RandomForestClassifier(n_estimators=100, max_depth=20,   random_state=42, n_jobs=-1)),
        ]
    elif model_name == "gbm":
        configs = [
            ("lr=0.05, n=100", GradientBoostingClassifier(learning_rate=0.05, n_estimators=100, random_state=42)),
            ("lr=0.10, n=100", GradientBoostingClassifier(learning_rate=0.10, n_estimators=100, random_state=42)),
            ("lr=0.10, n=200", GradientBoostingClassifier(learning_rate=0.10, n_estimators=200, random_state=42)),
            ("lr=0.20, n=100", GradientBoostingClassifier(learning_rate=0.20, n_estimators=100, random_state=42)),
            ("lr=0.10, d=3",   GradientBoostingClassifier(learning_rate=0.10, n_estimators=100, max_depth=3, random_state=42)),
        ]
    else:
        print(f"  Hyperparameter perturbation not implemented for {model_name}")
        return []

    results = []
    for label, clf in configs:
        scores = _score(clf, X_train, y_train, X_test, y_test)
        results.append((label, scores["roc_auc"], scores["f1"]))
        print(f"  {label:20s}  AUC={scores['roc_auc']:.4f}  F1={scores['f1']:.4f}")

    if output_dir:
        labels = [r[0] for r in results]
        aucs   = [r[1] for r in results]
        f1s    = [r[2] for r in results]
        x = np.arange(len(labels))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, vals, ylabel, color in [
            (axes[0], aucs, "ROC-AUC", "#1f77b4"),
            (axes[1], f1s,  "F1",      "#ff7f0e"),
        ]:
            ax.bar(x, vals, color=color, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(ylabel)
            ymin = min(vals) - 0.02
            ax.set_ylim(ymin, 1.0)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Hyperparameter Stability — {model_name}", fontsize=12)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"stability_hyperparameter_{model_name}.png")

    return results


def noise_stability(X_train, y_train, X_test, y_test,
                    model_name="gbm", noise_levels=None, seed=42, output_dir=None):
    """Add Gaussian noise scaled to feature std and check performance degradation."""
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20, 0.50]

    print(f"\n[Stability] Noise injection (model={model_name})")
    rng = np.random.default_rng(seed)
    feat_std = X_train.std(axis=0)
    results = []

    for sigma in noise_levels:
        if sigma == 0.0:
            X_tr_n, X_te_n = X_train.copy(), X_test.copy()
        else:
            X_tr_n = X_train + rng.normal(0, sigma * feat_std, X_train.shape).astype(np.float32)
            X_te_n = X_test  + rng.normal(0, sigma * feat_std, X_test.shape).astype(np.float32)
        scores = _score(_base_clf(model_name, seed), X_tr_n, y_train, X_te_n, y_test)
        results.append((sigma, scores["roc_auc"], scores["f1"]))
        print(f"  σ={sigma:.2f}  AUC={scores['roc_auc']:.4f}  F1={scores['f1']:.4f}")

    if output_dir:
        sigmas = [r[0] for r in results]
        aucs   = [r[1] for r in results]
        f1s    = [r[2] for r in results]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sigmas, aucs, "o-", color="#1f77b4", label="ROC-AUC")
        ax.plot(sigmas, f1s,  "s-", color="#ff7f0e", label="F1")
        ax.set_xlabel("Noise level σ (× feature std)")
        ax.set_ylabel("Score")
        ax.set_title(f"Noise Injection Stability — {model_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"stability_noise_{model_name}.png")

    return results


def _rbo(list1, list2, p=0.9):
    """Rank-Biased Overlap between two ranked lists (Webber et al. 2010)."""
    s, longer = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    d = len(longer)
    rbo = 0.0
    intersect = 0
    for k in range(1, d + 1):
        if k <= len(s) and longer[k-1] in s[:k]:
            intersect += 1
        if k <= len(s) and s[k-1] in longer[:k]:
            if longer[k-1] != s[k-1]:
                intersect += (s[k-1] in longer[:k])
        agreement = intersect / k
        rbo += agreement * (p ** (k - 1))
    return (1 - p) * rbo


def feature_importance_ranking_stability(X_train, y_train, X_test, y_test,
                                          feature_names, model_name="rf",
                                          n_runs=10, top_k=10, seed=42,
                                          output_dir=None):
    """Compute pairwise RBO across n_runs to check feature ranking stability."""
    print(f"\n[Stability] Feature importance ranking (RBO, model={model_name}, n_runs={n_runs})")
    rng = np.random.default_rng(seed)
    rankings = []

    for i in range(n_runs):
        s = int(rng.integers(1, 10000))
        clf = _base_clf(model_name, seed=s)
        clf.fit(X_train, y_train)

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=s)
            imp = r.importances_mean

        ranked = [feature_names[j] for j in np.argsort(imp)[::-1][:top_k]]
        rankings.append(ranked)

    rbos = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            rbos.append(_rbo(rankings[i], rankings[j]))
    rbos = np.array(rbos)

    print(f"  Pairwise RBO: mean={rbos.mean():.4f}  std={rbos.std():.4f}  "
          f"min={rbos.min():.4f}  (> 0.8 = stable)")

    from collections import Counter
    freq = Counter(f for r in rankings for f in r)
    top_feats = [f for f, _ in freq.most_common(top_k)]
    counts    = [freq[f] for f in top_feats]

    if output_dir:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # RBO distribution
        axes[0].hist(rbos, bins=20, color="#1f77b4", alpha=0.85, edgecolor="white")
        axes[0].axvline(rbos.mean(), color="black", linestyle="--",
                        label=f"mean={rbos.mean():.3f}")
        axes[0].set_xlabel("Pairwise RBO")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Feature Ranking Stability (RBO)\n{model_name}, n_runs={n_runs}")
        axes[0].legend(fontsize=9)
        axes[0].set_xlim(0, 1)

        # Feature frequency in top-k
        axes[1].barh(range(len(top_feats)), counts, color="#ff7f0e", alpha=0.85)
        axes[1].set_yticks(range(len(top_feats)))
        axes[1].set_yticklabels(top_feats, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel(f"# times in top-{top_k} (out of {n_runs} runs)")
        axes[1].set_title(f"Feature Consistency — {model_name}")
        axes[1].axvline(n_runs, color="red", linestyle=":", linewidth=0.8,
                        label=f"n_runs={n_runs}")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="x", alpha=0.3)

        fig.suptitle("PCS: Feature Importance Ranking Stability", fontsize=12)
        fig.tight_layout()
        _save_fig(fig, output_dir, f"stability_rbo_{model_name}.png")

    return {"rbo_mean": rbos.mean(), "rbo_std": rbos.std(),
            "top_features": top_feats, "frequency": counts}


def run_stability_analysis(split, model_name="gbm", n_bootstrap=30,
                           output_dir=None, seed=42):
    """Run all perturbation types on a split and return summary dict."""
    print(f"\n{'='*60}")
    print(f"  PCS Stability Analysis — model: {model_name}")
    print(f"{'='*60}")

    X_train_raw = getattr(split, "X_train_raw", split.X_train)
    X_test_raw  = getattr(split, "X_test_raw",  split.X_test)

    bs  = bootstrap_stability(split.X_train, split.y_train,
                              split.X_test,  split.y_test,
                              model_name=model_name, n_bootstrap=n_bootstrap,
                              seed=seed, output_dir=output_dir)

    pre = preprocessing_stability(X_train_raw, split.y_train,
                                   X_test_raw,  split.y_test,
                                   model_name=model_name, seed=seed,
                                   output_dir=output_dir)

    hyp = hyperparameter_stability(split.X_train, split.y_train,
                                    split.X_test,  split.y_test,
                                    model_name=model_name, output_dir=output_dir)

    noi = noise_stability(split.X_train, split.y_train,
                          split.X_test,  split.y_test,
                          model_name=model_name, seed=seed, output_dir=output_dir)

    rbo = feature_importance_ranking_stability(
        split.X_train, split.y_train,
        split.X_test,  split.y_test,
        feature_names=split.feature_names,
        model_name=model_name, n_runs=10,
        seed=seed, output_dir=output_dir,
    )

    print("\n[Stability Summary]")
    print(f"  Bootstrap AUC std : {bs['auc_std']:.4f}  (< 0.01 = stable)")
    print(f"  Bootstrap F1  std : {bs['f1_std']:.4f}  (< 0.02 = stable)")
    print(f"  Feature RBO mean  : {rbo['rbo_mean']:.4f}  (> 0.80 = stable)")

    return {"bootstrap": bs, "preprocessing": pre,
            "hyperparameter": hyp, "noise": noi, "rbo": rbo}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCS Stability Analysis")
    parser.add_argument("--data-dir",    default="../data")
    parser.add_argument("--output-dir",  default="../report/figures")
    parser.add_argument("--n-bootstrap", type=int, default=30)
    parser.add_argument("--model",       default="gbm",
                        choices=["logreg", "rf", "gbm"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--feature-mode", default="enhanced")
    parser.add_argument("--ae-embed-dir", default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from modeling import get_cross_validation_splits

    folds, _ = get_cross_validation_splits(
        data_dir=args.data_dir,
        max_samples_per_image=None,
        feature_mode=args.feature_mode,
        seed=args.seed,
        ae_embed_dir=args.ae_embed_dir,
    )
    split = folds[-1]["split"]

    run_stability_analysis(
        split,
        model_name=args.model,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
        seed=args.seed,
    )
