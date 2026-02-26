"""
experiments.py
--------------
Run the core experiments comparing SVM on raw TF-IDF vs. PCA-reduced data.
Collects accuracy, F1, and timing across different numbers of components.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pca_analysis import fit_pca, transform_pca
from svm_classifier import train_svm, evaluate_svm


def run_baseline(X_train, X_test, y_train, y_test) -> dict:
    """
    Experiment 1: SVM on raw TF-IDF (no dimensionality reduction).
    This is the control condition.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline SVM on Full TF-IDF")
    print("=" * 60)

    result = train_svm(X_train, y_train)
    metrics = evaluate_svm(result["model"], X_test, y_test)

    return {
        "model": result["model"],
        "metrics": metrics,
        "train_time": result["train_time"],
        "n_features": X_train.shape[1],
    }


def run_pca_sweep(
    X_train, X_test, y_train, y_test,
    component_counts: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Experiment 2: Train SVM at multiple PCA dimensionalities.

    For each n_components:
      1. Fit PCA on train, transform both train and test
      2. Train SVM on reduced data
      3. Record accuracy, F1, and timing

    Returns a DataFrame with one row per experiment.
    """
    if component_counts is None:
        component_counts = [5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    # Filter out components larger than the feature count
    max_features = X_train.shape[1]
    component_counts = [c for c in component_counts if c < max_features]

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: PCA Sweep — Varying Number of Components")
    print(f"  Components to test: {component_counts}")
    print("=" * 60)

    records = []

    for n_comp in component_counts:
        print(f"\n--- {n_comp} components ---")

        # PCA
        t0 = time.time()
        pca_model, X_train_pca = fit_pca(X_train, n_components=n_comp, seed=seed)
        X_test_pca = transform_pca(pca_model, X_test)
        pca_time = time.time() - t0

        variance_captured = np.sum(pca_model.explained_variance_ratio_)
        print(f"  Variance captured: {variance_captured:.4f}")

        # SVM
        svm_result = train_svm(X_train_pca, y_train)
        svm_metrics = evaluate_svm(svm_result["model"], X_test_pca, y_test)

        records.append({
            "n_components": n_comp,
            "accuracy": svm_metrics["accuracy"],
            "f1": svm_metrics["f1"],
            "precision": svm_metrics["precision"],
            "recall": svm_metrics["recall"],
            "variance_captured": variance_captured,
            "pca_time": pca_time,
            "svm_train_time": svm_result["train_time"],
            "svm_predict_time": svm_metrics["predict_time"],
            "total_time": pca_time + svm_result["train_time"],
            "confusion_matrix": svm_metrics["confusion_matrix"],
        })

    df = pd.DataFrame(records)
    print("\n" + "=" * 60)
    print("SWEEP RESULTS:")
    print(df[["n_components", "accuracy", "f1", "variance_captured",
              "total_time"]].to_string(index=False))
    print("=" * 60)

    return df


# ─── Experiment Visualization ────────────────────────────────────────────────


def plot_accuracy_vs_components(
    sweep_df: pd.DataFrame,
    baseline_accuracy: float,
    save_path: str = "results/accuracy_vs_components.png",
):
    """
    Key plot: accuracy as a function of number of PCA components,
    with the baseline (no PCA) as a reference line.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(sweep_df["n_components"], sweep_df["accuracy"],
            "o-", color="steelblue", linewidth=2.5, markersize=8, label="PCA + SVM")

    ax.axhline(baseline_accuracy, color="firebrick", linestyle="--", linewidth=2,
               label=f"Baseline SVM ({baseline_accuracy:.4f})")

    # Annotate each point
    for _, row in sweep_df.iterrows():
        ax.annotate(
            f"{row['accuracy']:.3f}",
            (row["n_components"], row["accuracy"]),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, color="steelblue",
        )

    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("SVM Accuracy vs. Number of PCA Components", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_f1_vs_components(
    sweep_df: pd.DataFrame,
    baseline_f1: float,
    save_path: str = "results/f1_vs_components.png",
):
    """F1 score as a function of number of PCA components."""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(sweep_df["n_components"], sweep_df["f1"],
            "s-", color="darkorange", linewidth=2.5, markersize=8, label="PCA + SVM")

    ax.axhline(baseline_f1, color="firebrick", linestyle="--", linewidth=2,
               label=f"Baseline SVM ({baseline_f1:.4f})")

    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Test F1 Score", fontsize=12)
    ax.set_title("SVM F1 Score vs. Number of PCA Components", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_time_vs_components(
    sweep_df: pd.DataFrame,
    baseline_train_time: float,
    save_path: str = "results/time_vs_components.png",
):
    """
    Stacked time comparison: PCA time + SVM training time vs. baseline.
    Shows the computational tradeoff.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(sweep_df))
    width = 0.6

    ax.bar(x, sweep_df["pca_time"], width, label="PCA Transform", color="lightcoral")
    ax.bar(x, sweep_df["svm_train_time"], width, bottom=sweep_df["pca_time"],
           label="SVM Training", color="steelblue")

    ax.axhline(baseline_train_time, color="firebrick", linestyle="--", linewidth=2,
               label=f"Baseline SVM time ({baseline_train_time:.2f}s)")

    ax.set_xticks(x)
    ax.set_xticklabels(sweep_df["n_components"].astype(int), fontsize=10)
    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Computation Time: PCA + SVM vs. Baseline", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_vs_variance(
    sweep_df: pd.DataFrame,
    baseline_accuracy: float,
    save_path: str = "results/accuracy_vs_variance.png",
):
    """
    Accuracy plotted against cumulative variance captured.
    Shows the relationship between information retained and classification quality.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(sweep_df["variance_captured"], sweep_df["accuracy"],
            "o-", color="steelblue", linewidth=2.5, markersize=8)

    ax.axhline(baseline_accuracy, color="firebrick", linestyle="--", linewidth=2,
               label=f"Baseline ({baseline_accuracy:.4f})")

    for _, row in sweep_df.iterrows():
        ax.annotate(
            f"{int(row['n_components'])} comp",
            (row["variance_captured"], row["accuracy"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, color="gray",
        )

    ax.set_xlabel("Cumulative Variance Captured", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Variance Retained by PCA", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_table(
    sweep_df: pd.DataFrame,
    baseline: dict,
    save_path: str = "results/summary_table.png",
):
    """
    Render a formatted summary table as an image (good for reports).
    Adds a final row summarizing the best PCA configuration relative to baseline.
    """
    # Build table data
    baseline_acc = baseline["metrics"]["accuracy"]
    baseline_f1 = baseline["metrics"]["f1"]

    rows = []
    rows.append({
        "Config": f"Baseline ({baseline['n_features']} feat.)",
        "Accuracy": f"{baseline_acc:.4f}",
        "F1": f"{baseline_f1:.4f}",
        "Variance": "100%",
        "Total Time (s)": f"{baseline['train_time']:.2f}",
    })

    # PCA configurations
    best_idx = sweep_df["accuracy"].idxmax()
    best_row = sweep_df.loc[best_idx]

    for _, row in sweep_df.iterrows():
        rows.append({
            "Config": f"PCA-{int(row['n_components'])}",
            "Accuracy": f"{row['accuracy']:.4f}",
            "F1": f"{row['f1']:.4f}",
            "Variance": f"{row['variance_captured']:.2%}",
            "Total Time (s)": f"{row['total_time']:.2f}",
        })

    # Summary row: highlight best PCA vs baseline (shows improvement over baseline)
    rows.append({
        "Config": f"Best PCA-{int(best_row['n_components'])} vs Baseline",
        "Accuracy": f"{best_row['accuracy']:.4f} (Δ {best_row['accuracy'] - baseline_acc:+.4f})",
        "F1": f"{best_row['f1']:.4f} (Δ {best_row['f1'] - baseline_f1:+.4f})",
        "Variance": f"{best_row['variance_captured']:.2%}",
        "Total Time (s)": f"{best_row['total_time']:.2f}",
    })

    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * len(rows) + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(table_df.columns)):
        table[0, j].set_facecolor("steelblue")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(table_df.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("Experiment Results Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
