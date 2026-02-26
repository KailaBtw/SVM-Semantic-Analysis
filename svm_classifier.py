"""
svm_classifier.py
-----------------
SVM training, evaluation, and visualization utilities.
Uses LinearSVC for efficiency on high-dimensional sparse data.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


# ─── Core SVM functions ─────────────────────────────────────────────────────


def train_svm(X_train, y_train, C: float = 1.0, seed: int = 42, max_iter: int = 5000) -> dict:
    """
    Train a linear SVM classifier. Returns the model and timing info.

    We use LinearSVC because:
      1. Uses liblinear, which is much faster for large sparse data
      2. Equivalent to SVC(kernel='linear') but scales O(n) instead of O(n^2)
      3. Perfect for the high-dimensional TF-IDF setting

    Returns:
        dict with keys: 'model', 'train_time', 'n_features', 'C'
    """
    print(f"  Training LinearSVC (C={C}, features={X_train.shape[1]})...")

    model = LinearSVC(C=C, random_state=seed, max_iter=max_iter, dual="auto")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"    Training time: {train_time:.2f}s")

    return {
        "model": model,
        "train_time": train_time,
        "n_features": X_train.shape[1],
        "C": C,
    }


def evaluate_svm(model, X_test, y_test) -> dict:
    """
    Evaluate SVM on test data. Returns a dict of all metrics.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0

    # Decision function values (distance to hyperplane, used for ROC)
    decision_vals = model.decision_function(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "decision_values": decision_vals,
        "predict_time": predict_time,
        "report": classification_report(y_test, y_pred, target_names=["Negative", "Positive"]),
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}  |  F1: {metrics['f1']:.4f}  "
          f"|  Predict time: {predict_time:.3f}s")

    return metrics


# ─── Visualization ───────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = "results/confusion_matrix.png",
):
    """Heatmap of confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax, annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix_comparison(
    cm_baseline: np.ndarray,
    cm_pca: np.ndarray,
    n_components: int,
    save_path: str = "results/confusion_matrix_comparison.png",
):
    """Side-by-side confusion matrices: baseline vs. best PCA config."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cm, title in [
        (axes[0], cm_baseline, "Baseline SVM\n(Full TF-IDF)"),
        (axes[1], cm_pca, f"PCA + SVM\n({n_components} components)"),
    ]:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=ax, annot_kws={"size": 16},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=13)

    plt.suptitle("Confusion Matrix Comparison", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_decision_boundary_2d(
    X_2d: np.ndarray,
    y: np.ndarray,
    model_2d=None,
    save_path: str = "results/svm_decision_boundary_2d.png",
):
    """
    Train a small SVM on 2D PCA-reduced data and visualize its decision boundary.
    This is a pedagogical plot — accuracy will be low in 2D, but it shows the concept.
    """
    # Train a small SVM on just 2 components if not provided
    if model_2d is None:
        model_2d = LinearSVC(C=1.0, random_state=42, max_iter=5000, dual="auto")
        model_2d.fit(X_2d, y)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mesh grid for decision boundary
    h = 0.5  # step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.15, cmap="RdYlGn")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.3)

    # Scatter points
    colors = {0: "salmon", 1: "mediumseagreen"}
    labels = {0: "Negative", 1: "Positive"}
    for label in [0, 1]:
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[label],
                   label=labels[label], alpha=0.3, s=8, edgecolors="none")

    # Draw the decision boundary line
    w = model_2d.coef_[0]
    b = model_2d.intercept_[0]
    x_boundary = np.linspace(x_min, x_max, 100)
    y_boundary = -(w[0] * x_boundary + b) / w[1]
    ax.plot(x_boundary, y_boundary, "k-", linewidth=2, label="Decision Boundary")

    # Margin lines
    y_margin_plus = -(w[0] * x_boundary + b - 1) / w[1]
    y_margin_minus = -(w[0] * x_boundary + b + 1) / w[1]
    ax.plot(x_boundary, y_margin_plus, "k--", linewidth=1, alpha=0.5, label="Margin")
    ax.plot(x_boundary, y_margin_minus, "k--", linewidth=1, alpha=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title("SVM Decision Boundary on 2D PCA Projection", fontsize=14)
    ax.legend(fontsize=10, markerscale=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_decision_value_distribution(
    decision_vals: np.ndarray,
    y: np.ndarray,
    title: str = "SVM Decision Value Distribution",
    save_path: str = "results/decision_value_dist.png",
):
    """
    Histogram of SVM decision function values by class.
    Shows how well the hyperplane separates the two classes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(decision_vals[y == 0], bins=80, alpha=0.6, color="salmon",
            label="Negative", density=True, edgecolor="white")
    ax.hist(decision_vals[y == 1], bins=80, alpha=0.6, color="mediumseagreen",
            label="Positive", density=True, edgecolor="white")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Decision Boundary")
    ax.set_xlabel("Decision Function Value (distance from hyperplane)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_svm_top_features(
    model,
    vectorizer,
    n_top: int = 25,
    save_path: str = "results/svm_top_features.png",
):
    """
    Show the words with highest and lowest SVM coefficients.
    Most positive = strongest indicator of positive sentiment.
    Most negative = strongest indicator of negative sentiment.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_pos_idx = coefs.argsort()[-n_top:][::-1]
    top_neg_idx = coefs.argsort()[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Positive sentiment features
    axes[0].barh(range(n_top), coefs[top_pos_idx], color="mediumseagreen", edgecolor="white")
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(feature_names[top_pos_idx], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("SVM Coefficient")
    axes[0].set_title(f"Top {n_top} Positive Sentiment Words", fontsize=12)

    # Negative sentiment features
    axes[1].barh(range(n_top), coefs[top_neg_idx], color="salmon", edgecolor="white")
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(feature_names[top_neg_idx], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("SVM Coefficient")
    axes[1].set_title(f"Top {n_top} Negative Sentiment Words", fontsize=12)

    plt.suptitle("SVM Feature Importance: Words Driving Sentiment Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
