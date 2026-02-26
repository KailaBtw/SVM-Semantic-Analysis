"""
preprocessing.py
-----------------
Convert raw text into TF-IDF feature matrices.
Includes diagnostics and visualizations of the feature space.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


def build_tfidf(
    X_train_text: list[str],
    X_test_text: list[str],
    max_features: int = 10_000,
    min_df: int = 5,
    max_df: float = 0.95,
    ngram_range: tuple = (1, 1),
) -> tuple:
    """
    Fit TF-IDF on training text, transform both train and test.

    Returns:
        X_train_tfidf: sparse matrix (n_train, max_features)
        X_test_tfidf:  sparse matrix (n_test, max_features)
        vectorizer:    fitted TfidfVectorizer
    """
    print(f"\nBuilding TF-IDF (max_features={max_features}, "
          f"min_df={min_df}, max_df={max_df}, ngrams={ngram_range})...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        ngram_range=ngram_range,
        sublinear_tf=True,       # apply log(1 + tf), common for text
        dtype=np.float64,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    vocab_size = len(vectorizer.vocabulary_)
    sparsity = 1.0 - (X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))

    print(f"  Vocabulary size:  {vocab_size:,}")
    print(f"  Train matrix:     {X_train_tfidf.shape}")
    print(f"  Test matrix:      {X_test_tfidf.shape}")
    print(f"  Sparsity:         {sparsity:.4%}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ─── Visualization helpers ───────────────────────────────────────────────────


def plot_tfidf_sparsity(X_tfidf, save_path: str = "results/tfidf_sparsity.png"):
    """Visualize the sparsity pattern of TF-IDF matrix (small sample)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sparsity heatmap of a small slice
    sample = X_tfidf[:100, :200]
    if issparse(sample):
        sample = sample.toarray()

    axes[0].imshow(sample != 0, cmap="Blues", aspect="auto", interpolation="none")
    axes[0].set_title("TF-IDF Non-Zero Pattern (100 docs × 200 features)", fontsize=11)
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Document Index")

    # Right: histogram of non-zero values per document
    if issparse(X_tfidf):
        nnz_per_doc = np.diff(X_tfidf.indptr)
    else:
        nnz_per_doc = np.count_nonzero(X_tfidf, axis=1)

    axes[1].hist(nnz_per_doc, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].set_title("Non-Zero Features per Document", fontsize=11)
    axes[1].set_xlabel("Number of Non-Zero Features")
    axes[1].set_ylabel("Number of Documents")
    axes[1].axvline(np.mean(nnz_per_doc), color="red", linestyle="--",
                    label=f"Mean = {np.mean(nnz_per_doc):.0f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_top_tfidf_terms(
    vectorizer,
    X_tfidf,
    y,
    top_n: int = 20,
    save_path: str = "results/top_tfidf_terms.png",
):
    """Bar chart of highest mean TF-IDF terms for positive vs negative reviews."""
    feature_names = np.array(vectorizer.get_feature_names_out())

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (label, label_name, color) in enumerate(
        [(0, "Negative", "salmon"), (1, "Positive", "mediumseagreen")]
    ):
        mask = y == label
        subset = X_tfidf[mask]
        mean_tfidf = np.asarray(subset.mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[-top_n:][::-1]

        axes[idx].barh(
            range(top_n), mean_tfidf[top_idx], color=color, edgecolor="white"
        )
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels(feature_names[top_idx], fontsize=9)
        axes[idx].invert_yaxis()
        axes[idx].set_xlabel("Mean TF-IDF Score")
        axes[idx].set_title(f"Top {top_n} Terms — {label_name} Reviews", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_document_length_distribution(
    X_train_text: list[str],
    y_train: np.ndarray,
    save_path: str = "results/doc_length_dist.png",
):
    """Histogram of review lengths by sentiment class."""
    lengths_neg = [len(t.split()) for t, y in zip(X_train_text, y_train) if y == 0]
    lengths_pos = [len(t.split()) for t, y in zip(X_train_text, y_train) if y == 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths_neg, bins=60, alpha=0.6, color="salmon", label="Negative", edgecolor="white")
    ax.hist(lengths_pos, bins=60, alpha=0.6, color="mediumseagreen", label="Positive", edgecolor="white")
    ax.set_xlabel("Review Length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Review Lengths by Sentiment")
    ax.legend()
    ax.set_xlim(0, 1500)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
