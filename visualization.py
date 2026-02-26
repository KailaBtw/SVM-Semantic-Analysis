"""
visualization.py
-----------------
Additional report-quality visualizations:
  - Word clouds
  - Dimensionality comparison dashboard
  - Eigenvalue spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_wordclouds(
    vectorizer,
    X_tfidf,
    y,
    save_path: str = "results/wordclouds.png",
):
    """
    Side-by-side word clouds for positive and negative reviews,
    weighted by mean TF-IDF score.
    """
    feature_names = vectorizer.get_feature_names_out()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for idx, (label, title, colormap) in enumerate([
        (0, "Negative Reviews", "Reds"),
        (1, "Positive Reviews", "Greens"),
    ]):
        mask = y == label
        mean_tfidf = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
        word_weights = {feature_names[i]: mean_tfidf[i] for i in range(len(feature_names))}

        wc = WordCloud(
            width=800, height=500,
            background_color="white",
            colormap=colormap,
            max_words=150,
            prefer_horizontal=0.7,
        ).generate_from_frequencies(word_weights)

        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].set_title(title, fontsize=16, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("Word Clouds by Sentiment (TF-IDF Weighted)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_eigenvalue_spectrum(
    model,
    save_path: str = "results/eigenvalue_spectrum.png",
):
    """
    Log-scale plot of singular values (eigenvalue proxy) to show
    the rapid decay typical of text data.
    """
    singular_values = model.singular_values_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    axes[0].plot(range(1, len(singular_values) + 1), singular_values,
                 color="steelblue", linewidth=2)
    axes[0].set_xlabel("Component Index", fontsize=11)
    axes[0].set_ylabel("Singular Value", fontsize=11)
    axes[0].set_title("Singular Value Spectrum (Linear)", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(range(1, len(singular_values) + 1), singular_values,
                     color="steelblue", linewidth=2)
    axes[1].set_xlabel("Component Index", fontsize=11)
    axes[1].set_ylabel("Singular Value (log scale)", fontsize=11)
    axes[1].set_title("Singular Value Spectrum (Log)", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Singular Value Decay of TF-IDF Matrix", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_component_variance_heatmap(
    model,
    n_show: int = 20,
    save_path: str = "results/component_variance_heatmap.png",
):
    """
    Heatmap showing explained variance for first N components.
    Each cell is one component with color intensity = variance ratio.
    """
    var_ratios = model.explained_variance_ratio_[:n_show]

    fig, ax = plt.subplots(figsize=(14, 3))

    data = var_ratios.reshape(1, -1)
    sns.heatmap(
        data, ax=ax, cmap="YlOrRd",
        annot=True, fmt=".3f", annot_kws={"size": 9},
        xticklabels=[f"PC{i+1}" for i in range(n_show)],
        yticklabels=["Var. Ratio"],
        cbar_kws={"label": "Explained Variance Ratio"},
    )
    ax.set_title(f"Explained Variance per Component (First {n_show})", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_radar(
    baseline_metrics: dict,
    pca_metrics: dict,
    n_components: int,
    save_path: str = "results/metrics_radar.png",
):
    """
    Radar / spider chart comparing baseline vs PCA across multiple metrics.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1"]
    baseline_vals = [baseline_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]
    pca_vals = [pca_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]

    # Close the radar
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    baseline_vals += baseline_vals[:1]
    pca_vals += pca_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, baseline_vals, "o-", linewidth=2, label="Baseline SVM", color="firebrick")
    ax.fill(angles, baseline_vals, alpha=0.15, color="firebrick")

    ax.plot(angles, pca_vals, "s-", linewidth=2,
            label=f"PCA-{n_components} + SVM", color="steelblue")
    ax.fill(angles, pca_vals, alpha=0.15, color="steelblue")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Classification Metrics Comparison", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dimensionality_dashboard(
    sweep_df,
    baseline_accuracy: float,
    baseline_time: float,
    save_path: str = "results/dimensionality_dashboard.png",
):
    """
    Multi-panel dashboard showing the full picture:
      - Accuracy vs components
      - Time vs components
      - Accuracy vs variance captured
      - Accuracy vs time (Pareto front)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Accuracy vs Components
    ax = axes[0, 0]
    ax.plot(sweep_df["n_components"], sweep_df["accuracy"], "o-",
            color="steelblue", linewidth=2, markersize=7)
    ax.axhline(baseline_accuracy, color="firebrick", linestyle="--", linewidth=2)
    ax.set_xlabel("PCA Components")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Components")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 2: Time vs Components
    ax = axes[0, 1]
    ax.bar(range(len(sweep_df)), sweep_df["total_time"], color="steelblue", alpha=0.8)
    ax.axhline(baseline_time, color="firebrick", linestyle="--", linewidth=2)
    ax.set_xticks(range(len(sweep_df)))
    ax.set_xticklabels(sweep_df["n_components"].astype(int), fontsize=9, rotation=45)
    ax.set_xlabel("PCA Components")
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Computation Time vs. Components")
    ax.grid(True, alpha=0.2, axis="y")

    # Panel 3: Accuracy vs Variance
    ax = axes[1, 0]
    ax.plot(sweep_df["variance_captured"], sweep_df["accuracy"], "o-",
            color="darkorange", linewidth=2, markersize=7)
    ax.axhline(baseline_accuracy, color="firebrick", linestyle="--", linewidth=2)
    ax.set_xlabel("Variance Captured")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Variance Retained")
    ax.grid(True, alpha=0.3)

    # Panel 4: Accuracy vs Time (Pareto)
    ax = axes[1, 1]
    ax.scatter(sweep_df["total_time"], sweep_df["accuracy"],
               c="steelblue", s=80, zorder=5, edgecolors="white")
    for _, row in sweep_df.iterrows():
        ax.annotate(f"{int(row['n_components'])}",
                    (row["total_time"], row["accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")
    ax.scatter([baseline_time], [baseline_accuracy], c="firebrick", s=100,
               marker="*", zorder=5, label="Baseline")
    ax.set_xlabel("Total Time (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Time (Pareto View)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Dimensionality Reduction Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
