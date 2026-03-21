"""Copula-based tail dependence analysis for radiance and expert features."""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import FEATURE_COLS  # noqa: E402

LABELED_FILES = ("O012791.npz", "O013257.npz", "O013490.npz")
RADIANCE_NAMES = ["Rad_DF", "Rad_CF", "Rad_BF", "Rad_AF", "Rad_AN"]
EXPERT_NAMES = ["NDAI", "SD", "CORR"]


def load_labeled_data(data_dir):
    """Load labeled data, return (features_dict, labels)."""
    all_feats = []
    all_labels = []
    for fname in LABELED_FILES:
        fp = os.path.join(data_dir, fname)
        arr = np.load(fp)
        key = list(arr.files)[0]
        data = arr[key]
        if data.shape[1] != 11:
            continue
        feats = data[:, 2:10].astype(np.float64)
        lbl = data[:, 10].astype(int)
        mask = lbl != 0
        all_feats.append(feats[mask])
        all_labels.append(lbl[mask])

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    y_bin = (y == 1).astype(int)  # 1=cloud, 0=no-cloud

    feat_dict = {}
    for i, name in enumerate(FEATURE_COLS):
        feat_dict[name] = X[:, i]
    return feat_dict, y_bin


def to_pseudo_obs(u):
    """Rank-transform to pseudo-observations in (0, 1)."""
    n = len(u)
    ranks = stats.rankdata(u)
    return ranks / (n + 1)


def gaussian_copula_kendall_tau(rho):
    """tau = (2/pi) * arcsin(rho)."""
    return (2.0 / np.pi) * np.arcsin(rho)


def gaussian_copula_fit(u, v):
    """Estimate rho from Kendall's tau."""
    tau, _ = stats.kendalltau(u, v)
    rho = np.sin(np.pi * tau / 2.0)
    rho = np.clip(rho, -0.999, 0.999)
    return {"rho": rho, "kendall_tau": tau, "lambda_L": 0.0, "lambda_U": 0.0}


def clayton_nll(theta, u, v):
    """NLL for Clayton copula: C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}."""
    if theta <= 0:
        return 1e12
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)

    A = u ** (-theta) + v ** (-theta) - 1.0
    A = np.clip(A, eps, None)

    log_density = (
        np.log(1 + theta)
        - (1 + theta) * (np.log(u) + np.log(v))
        - (2 + 1.0 / theta) * np.log(A)
    )
    return -np.sum(log_density)


def clayton_copula_fit(u, v):
    """Fit Clayton copula by MLE, return theta and lambda_L."""
    tau_emp, _ = stats.kendalltau(u, v)

    result = minimize_scalar(
        lambda t: clayton_nll(t, u, v),
        bounds=(0.01, 50.0),
        method="bounded",
    )
    theta = result.x

    # lambda_L = 2^{-1/theta}
    lambda_L = 2.0 ** (-1.0 / theta) if theta > 0 else 0.0
    tau_model = theta / (theta + 2.0)

    return {
        "theta": theta,
        "kendall_tau_empirical": tau_emp,
        "kendall_tau_model": tau_model,
        "lambda_L": lambda_L,
        "lambda_U": 0.0,
    }


def analyze_radiance_pairs(feat_dict, labels, output_dir):
    """Pairwise copula analysis on radiance angles, cloud vs non-cloud."""
    cloud_mask = labels == 1
    nocloud_mask = labels == 0

    pairs = []
    for i in range(len(RADIANCE_NAMES)):
        for j in range(i + 1, len(RADIANCE_NAMES)):
            pairs.append((RADIANCE_NAMES[i], RADIANCE_NAMES[j]))

    results_cloud = []
    results_nocloud = []

    for a, b in pairs:
        for mask, label, results_list in [
            (cloud_mask, "cloud", results_cloud),
            (nocloud_mask, "no-cloud", results_nocloud),
        ]:
            u = to_pseudo_obs(feat_dict[a][mask])
            v = to_pseudo_obs(feat_dict[b][mask])

            gauss = gaussian_copula_fit(u, v)
            clay = clayton_copula_fit(u, v)
            results_list.append({
                "pair": f"{a.replace('Rad_','')}-{b.replace('Rad_','')}",
                "gauss_tau": gauss["kendall_tau"],
                "gauss_rho": gauss["rho"],
                "clay_theta": clay["theta"],
                "clay_tau_model": clay["kendall_tau_model"],
                "clay_lambda_L": clay["lambda_L"],
                "tau_emp": clay["kendall_tau_empirical"],
            })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pair_labels = [r["pair"] for r in results_cloud]
    x = np.arange(len(pair_labels))
    w = 0.35

    ax = axes[0]
    tau_cloud = [r["tau_emp"] for r in results_cloud]
    tau_nocloud = [r["tau_emp"] for r in results_nocloud]
    ax.bar(x - w / 2, tau_cloud, w, label="Cloud", color="#e74c3c", alpha=0.8)
    ax.bar(x + w / 2, tau_nocloud, w, label="No-cloud", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Empirical Kendall's τ by Angle Pair")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)

    ax = axes[1]
    lam_cloud = [r["clay_lambda_L"] for r in results_cloud]
    lam_nocloud = [r["clay_lambda_L"] for r in results_nocloud]
    ax.bar(x - w / 2, lam_cloud, w, label="Cloud", color="#e74c3c", alpha=0.8)
    ax.bar(x + w / 2, lam_nocloud, w, label="No-cloud", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("λ_L (Clayton)")
    ax.set_title("Lower Tail Dependence (Clayton Copula)")
    ax.legend()

    fig.suptitle("Copula Analysis: Multi-Angle Radiance Dependence Structure", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "copula_radiance_tail_dependence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [
        (axes[0], results_cloud, "Cloud Pixels"),
        (axes[1], results_nocloud, "No-Cloud Pixels"),
    ]:
        tau_gauss = [r["gauss_tau"] for r in results]
        tau_clay_model = [r["clay_tau_model"] for r in results]
        tau_emp = [r["tau_emp"] for r in results]

        ax.plot(x, tau_emp, "ko-", label="Empirical τ", ms=6)
        ax.plot(x, tau_gauss, "s--", color="#2ecc71", label="Gaussian τ", ms=5)
        ax.plot(x, tau_clay_model, "^--", color="#e67e22", label="Clayton τ (model)", ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Kendall's τ")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Gaussian vs Clayton Copula: Model Fit Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "copula_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return results_cloud, results_nocloud


def analyze_expert_features(feat_dict, labels, output_dir):
    """Copula scatter plots for NDAI, CORR, SD pairs."""
    cloud_mask = labels == 1
    nocloud_mask = labels == 0

    expert_pairs = [("NDAI", "SD"), ("NDAI", "CORR"), ("SD", "CORR")]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, (a, b) in enumerate(expert_pairs):
        for row, (mask, group_label, cmap) in enumerate([
            (cloud_mask, "Cloud", "Reds"),
            (nocloud_mask, "No-Cloud", "Blues"),
        ]):
            ax = axes[row, col]
            u_raw = feat_dict[a][mask]
            v_raw = feat_dict[b][mask]

            n = len(u_raw)
            if n > 5000:
                idx = np.random.default_rng(42).choice(n, 5000, replace=False)
                u_plot, v_plot = u_raw[idx], v_raw[idx]
            else:
                u_plot, v_plot = u_raw, v_raw

            u_pseudo = to_pseudo_obs(u_plot)
            v_pseudo = to_pseudo_obs(v_plot)

            ax.scatter(u_pseudo, v_pseudo, s=1, alpha=0.3,
                       c=u_pseudo + v_pseudo, cmap=cmap)
            ax.set_xlabel(f"{a} (pseudo-obs)")
            ax.set_ylabel(f"{b} (pseudo-obs)")
            ax.set_title(f"{group_label}: {a} vs {b}")

            u = to_pseudo_obs(u_raw)
            v = to_pseudo_obs(v_raw)
            gauss = gaussian_copula_fit(u, v)
            clay = clayton_copula_fit(u, v)
            ax.text(
                0.05, 0.95,
                f"τ={clay['kendall_tau_empirical']:.3f}\n"
                f"ρ_G={gauss['rho']:.3f}\n"
                f"θ_C={clay['theta']:.2f}\n"
                f"λ_L={clay['lambda_L']:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

    fig.suptitle("Expert Feature Joint Structure: Copula Pseudo-Observation Scatter", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "copula_expert_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(results_cloud, results_nocloud):
    """Print copula summary table."""
    print("\n" + "=" * 80)
    print("COPULA ANALYSIS SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Pair':<12} | {'Group':<10} | {'τ_emp':>7} | {'ρ_Gauss':>8} | {'θ_Clay':>7} | {'λ_L':>6}"
    print(header)
    print("-" * len(header))
    for rc, rn in zip(results_cloud, results_nocloud):
        print(f"{rc['pair']:<12} | {'Cloud':<10} | {rc['tau_emp']:>7.3f} | {rc['gauss_rho']:>8.3f} | {rc['clay_theta']:>7.2f} | {rc['clay_lambda_L']:>6.3f}")
        print(f"{rn['pair']:<12} | {'No-Cloud':<10} | {rn['tau_emp']:>7.3f} | {rn['gauss_rho']:>8.3f} | {rn['clay_theta']:>7.2f} | {rn['clay_lambda_L']:>6.3f}")
    print("=" * 80)
    print("\nKey findings:")
    avg_lam_cloud = np.mean([r["clay_lambda_L"] for r in results_cloud])
    avg_lam_nocloud = np.mean([r["clay_lambda_L"] for r in results_nocloud])
    print(f"  Mean λ_L (Cloud):    {avg_lam_cloud:.3f}")
    print(f"  Mean λ_L (No-Cloud): {avg_lam_nocloud:.3f}")
    if avg_lam_cloud > avg_lam_nocloud:
        print("  → Cloud pixels show STRONGER lower tail dependence across angles.")
    else:
        print("  → No-cloud pixels show STRONGER lower tail dependence across angles.")
    print("  → This suggests that extreme low-radiance events are more correlated for one group,")
    print("    providing evidence beyond linear correlation for discriminating cloud/no-cloud.")


def main():
    parser = argparse.ArgumentParser(description="Copula tail dependence analysis")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="../report/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading labeled data...")
    feat_dict, labels = load_labeled_data(args.data_dir)
    n_cloud = labels.sum()
    n_nocloud = len(labels) - n_cloud
    print(f"  Cloud: {n_cloud}, No-cloud: {n_nocloud}")

    print("\n[1/2] Multi-angle radiance copula analysis...")
    rc, rn = analyze_radiance_pairs(feat_dict, labels, args.output_dir)
    print_summary_table(rc, rn)

    print("\n[2/2] Expert feature joint structure analysis...")
    analyze_expert_features(feat_dict, labels, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
