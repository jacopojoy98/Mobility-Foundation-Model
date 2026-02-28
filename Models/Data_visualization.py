import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


# =========================
# Utility
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# 1. Marginal Histograms
# =========================

def plot_marginal_histograms(X, bins=30, outdir="figures"):
    ensure_dir(outdir)
    D = X.shape[1]

    cols = int(np.ceil(np.sqrt(D)))
    rows = int(np.ceil(D / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for d in range(D):
        axes[d].hist(X[:, d], bins=bins)
        axes[d].set_title(f"Dim {d}")
    for ax in axes[D:]:
        ax.axis("off")

    fig.suptitle("Marginal Histograms")
    fig.tight_layout()
    fig.savefig(f"{outdir}/marginal_histograms.png", dpi=150)
    plt.close(fig)


# =========================
# 2. Marginal Boxplots
# =========================

def plot_marginal_boxplots(X, outdir="figures"):
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(max(8, X.shape[1] * 0.4), 5))
    ax.boxplot(X, showfliers=False)
    ax.set_xlabel("Dimension")
    ax.set_title("Marginal Boxplots")

    fig.tight_layout()
    fig.savefig(f"{outdir}/marginal_boxplots.png", dpi=150)
    plt.close(fig)


# =========================
# 3. Covariance & Correlation
# =========================

def plot_matrix(M, title, filename, outdir="figures"):
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(f"{outdir}/{filename}", dpi=150)
    plt.close(fig)


def plot_cov_corr(X, outdir="figures"):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    corr = np.corrcoef(Xc, rowvar=False)

    plot_matrix(cov, "Covariance Matrix", "covariance_matrix.png", outdir)
    plot_matrix(corr, "Correlation Matrix", "correlation_matrix.png", outdir)


# =========================
# 4. PCA Diagnostics
# =========================

def plot_pca(X, outdir="figures"):
    ensure_dir(outdir)
    Xc = X - X.mean(axis=0)
    N = X.shape[0]

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (N - 1)
    var_ratio = var / var.sum()
    cum_var = np.cumsum(var_ratio)

    # Scree plot
    fig, ax = plt.subplots()
    ax.plot(var_ratio, marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    fig.tight_layout()
    fig.savefig(f"{outdir}/pca_scree.png", dpi=150)
    plt.close(fig)

    # Cumulative variance
    fig, ax = plt.subplots()
    ax.plot(cum_var, marker="o")
    ax.axhline(0.95, linestyle="--")
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative Variance")
    ax.set_title("PCA Cumulative Variance")
    fig.tight_layout()
    fig.savefig(f"{outdir}/pca_cumulative_variance.png", dpi=150)
    plt.close(fig)

    # First two PCs
    if Vt.shape[0] >= 2:
        Z = Xc @ Vt[:2].T
        fig, ax = plt.subplots()
        ax.scatter(Z[:, 0], Z[:, 1], s=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Projection onto First Two PCs")
        fig.tight_layout()
        fig.savefig(f"{outdir}/pca_projection_pc1_pc2.png", dpi=150)
        plt.close(fig)


# =========================
# 5. Norm Distribution
# =========================

def plot_norm_distribution(X, outdir="figures"):
    ensure_dir(outdir)
    norms = np.linalg.norm(X, axis=1)

    fig, ax = plt.subplots()
    ax.hist(norms, bins=30)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.set_title("Norm Distribution")

    fig.tight_layout()
    fig.savefig(f"{outdir}/norm_distribution.png", dpi=150)
    plt.close(fig)


# =========================
# 6. Pairwise Distance Distribution
# =========================

def plot_pairwise_distances(X, max_samples=2000, outdir="figures"):
    ensure_dir(outdir)

    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    diffs = X[:, None, :] - X[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists = dists[np.triu_indices_from(dists, k=1)]

    fig, ax = plt.subplots()
    ax.hist(dists, bins=30)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Distance Distribution")

    fig.tight_layout()
    fig.savefig(f"{outdir}/pairwise_distances.png", dpi=150)
    plt.close(fig)


# =========================
# 7. Master Function
# =========================

def visualize_distribution(X, outdir="figures"):
    if len(X[0]) <20:
        print("Warning: High-dimensional data may produce very large plots. Consider using dimensionality reduction before visualization.")
        plot_norm_distribution(X, outdir=outdir)
    plot_marginal_histograms(X, outdir=outdir)
    plot_marginal_boxplots(X, outdir=outdir)
    plot_cov_corr(X, outdir=outdir)
    plot_pca(X, outdir=outdir)
    plot_pairwise_distances(X, outdir=outdir)

if __name__ == "__main__":
    with open('Data/Mobility/trajectory_tokens_split_0.pkl', 'rb') as f:
        train_trajectories = pickle.load(f) 
    with open('Data/Mobility/visit_tokens_split_0.pkl', 'rb') as f:
        train_visits = pickle.load(f)

    X = []
    for user_trajectories in train_trajectories:
        for traj_token in user_trajectories:
            for point in traj_token:
                X.append(point)
    X = np.array(X)
    visualize_distribution(X, outdir="figures/train_trajectory_tokens")
