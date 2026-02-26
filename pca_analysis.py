"""
pca_analysis.py
---------------
PCA / TruncatedSVD for dimensionality reduction on TF-IDF text data.
Includes scree plots, cumulative variance, 2D/3D projections, and component inspection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D


# ─── Core PCA functions ─────────────────────────────────────────────────────


def fit_pca(X_train, n_components: int, seed: int = 42) -> tuple:
    """
    Fit TruncatedSVD (sparse-friendly PCA) on training data.

    We use TruncatedSVD because:
      1. It works directly on sparse TF-IDF matrices (no dense conversion needed)
      2. Mathematically equivalent to PCA on centered data
      3. Much more memory-efficient for large vocabularies

    Returns:
        model:          fitted TruncatedSVD
        X_train_reduced: np.ndarray (n_train, n_components)
    """
    model = TruncatedSVD(n_components=n_components, random_state=seed)
    X_train_reduced = model.fit_transform(X_train)
    return model, X_train_reduced


def transform_pca(model, X):
    """Transform data using a fitted PCA/SVD model."""
    return model.transform(X)


def get_explained_variance(model) -> tuple:
    """
    Return per-component and cumulative explained variance ratios.

    Returns:
        individual: np.ndarray — variance ratio per component
        cumulative: np.ndarray — cumulative variance ratio
    """
    individual = model.explained_variance_ratio_
    cumulative = np.cumsum(individual)
    return individual, cumulative


def find_n_components_for_variance(model, thresholds: list = [0.80, 0.90, 0.95, 0.99]) -> dict:
    """
    Find how many components are needed to reach each variance threshold.

    Returns:
        dict mapping threshold -> n_components needed
    """
    _, cumulative = get_explained_variance(model)
    result = {}
    for t in thresholds:
        idx = np.searchsorted(cumulative, t) + 1
        result[t] = min(idx, len(cumulative))
    return result


# ─── Visualization ───────────────────────────────────────────────────────────


def plot_scree(model, save_path: str = "results/scree_plot.png"):
    """
    Scree plot: individual + cumulative explained variance.
    This is one of the key figures for your paper.
    """
    individual, cumulative = get_explained_variance(model)
    n = len(individual)
    components = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for individual variance
    color1 = "steelblue"
    ax1.bar(components, individual, color=color1, alpha=0.7, label="Individual", width=0.8)
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Line for cumulative variance on secondary axis
    ax2 = ax1.twinx()
    color2 = "firebrick"
    ax2.plot(components, cumulative, color=color2, linewidth=2.5, marker="o",
             markersize=3, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance Ratio", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Reference lines at common thresholds
    for thresh, style in [(0.90, "--"), (0.95, ":"), (0.99, "-.")]:
        n_comp = np.searchsorted(cumulative, thresh) + 1
        if n_comp <= n:
            ax2.axhline(thresh, color="gray", linestyle=style, alpha=0.5)
            ax2.annotate(
                f"{thresh:.0%} at {n_comp} components",
                xy=(n_comp, thresh),
                xytext=(n_comp + n * 0.05, thresh - 0.03),
                fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    ax1.set_title("Scree Plot: Explained Variance by Principal Component", fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cumulative_variance_zoom(model, save_path: str = "results/cumulative_variance.png"):
    """
    Focused cumulative variance plot — useful for showing the 'elbow'.
    """
    _, cumulative = get_explained_variance(model)
    n = len(cumulative)
    components = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(components, cumulative, color="steelblue", linewidth=2.5)
    ax.fill_between(components, cumulative, alpha=0.15, color="steelblue")

    # Mark thresholds
    for thresh, color in [(0.80, "green"), (0.90, "orange"), (0.95, "red"), (0.99, "darkred")]:
        n_comp = np.searchsorted(cumulative, thresh) + 1
        if n_comp <= n:
            ax.axhline(thresh, color=color, linestyle="--", alpha=0.4, linewidth=1)
            ax.axvline(n_comp, color=color, linestyle="--", alpha=0.4, linewidth=1)
            ax.scatter([n_comp], [cumulative[n_comp - 1]], color=color, s=60, zorder=5)
            ax.annotate(
                f"{thresh:.0%} → {n_comp} comp.",
                xy=(n_comp, cumulative[n_comp - 1]),
                xytext=(n_comp + n * 0.04, cumulative[n_comp - 1] - 0.04),
                fontsize=9, color=color, fontweight="bold",
            )

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("Cumulative Variance Explained vs. Number of Components", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_2d_projection(
    X_reduced_2d: np.ndarray,
    y: np.ndarray,
    save_path: str = "results/pca_2d_projection.png",
):
    """
    Scatter plot of data projected onto first 2 principal components.
    Color by sentiment label. Shows how well classes separate in 2D.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {0: "salmon", 1: "mediumseagreen"}
    labels = {0: "Negative", 1: "Positive"}

    for label in [0, 1]:
        mask = y == label
        ax.scatter(
            X_reduced_2d[mask, 0],
            X_reduced_2d[mask, 1],
            c=colors[label],
            label=labels[label],
            alpha=0.25,
            s=8,
            edgecolors="none",
        )

    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title("IMDB Reviews Projected onto First 2 Principal Components", fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_3d_projection(
    X_reduced_3d: np.ndarray,
    y: np.ndarray,
    save_path: str = "results/pca_3d_projection.png",
):
    """
    3D scatter of data projected onto first 3 principal components.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    colors = {0: "salmon", 1: "mediumseagreen"}
    labels = {0: "Negative", 1: "Positive"}

    for label in [0, 1]:
        mask = y == label
        ax.scatter(
            X_reduced_3d[mask, 0],
            X_reduced_3d[mask, 1],
            X_reduced_3d[mask, 2],
            c=colors[label],
            label=labels[label],
            alpha=0.2,
            s=6,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title("3D PCA Projection of IMDB Reviews", fontsize=14)
    ax.legend(markerscale=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_component_top_words(
    model,
    vectorizer,
    n_components_to_show: int = 6,
    n_words: int = 15,
    save_path: str = "results/pca_component_words.png",
):
    """
    Show the top contributing words for each of the first N principal components.
    This helps interpret what each component is 'about'.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    n_show = min(n_components_to_show, model.n_components)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(n_show):
        component = model.components_[i]

        # Top positive and negative loadings
        top_pos_idx = component.argsort()[-n_words:][::-1]
        top_neg_idx = component.argsort()[:n_words]

        combined_idx = np.concatenate([top_neg_idx, top_pos_idx])
        combined_vals = component[combined_idx]
        combined_words = feature_names[combined_idx]

        colors = ["salmon" if v < 0 else "steelblue" for v in combined_vals]

        axes[i].barh(range(len(combined_vals)), combined_vals, color=colors)
        axes[i].set_yticks(range(len(combined_words)))
        axes[i].set_yticklabels(combined_words, fontsize=8)
        axes[i].set_title(f"PC {i + 1} ({model.explained_variance_ratio_[i]:.2%} var.)",
                          fontsize=11)
        axes[i].axvline(0, color="black", linewidth=0.5)

    # Hide unused axes
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Words by Loading in First Principal Components", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pairwise_components(
    X_reduced: np.ndarray,
    y: np.ndarray,
    pairs: list = [(0, 1), (0, 2), (1, 2), (0, 3)],
    save_path: str = "results/pca_pairwise.png",
):
    """
    Grid of scatter plots for different pairs of principal components.
    Shows separation from multiple angles.
    """
    n_pairs = len(pairs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {0: "salmon", 1: "mediumseagreen"}
    labels = {0: "Negative", 1: "Positive"}

    for idx, (i, j) in enumerate(pairs):
        if idx >= len(axes):
            break
        for label in [0, 1]:
            mask = y == label
            axes[idx].scatter(
                X_reduced[mask, i], X_reduced[mask, j],
                c=colors[label], label=labels[label],
                alpha=0.2, s=6, edgecolors="none",
            )
        axes[idx].set_xlabel(f"PC {i + 1}", fontsize=10)
        axes[idx].set_ylabel(f"PC {j + 1}", fontsize=10)
        axes[idx].set_title(f"PC {i + 1} vs PC {j + 1}", fontsize=11)
        axes[idx].legend(fontsize=9, markerscale=3)
        axes[idx].grid(True, alpha=0.2)

    plt.suptitle("Pairwise PCA Component Projections", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
