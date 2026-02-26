"""
data_loader.py
--------------
Load the IMDB 50k review dataset and return train/test splits.
Uses HuggingFace datasets library for convenience; falls back to manual download.
"""

import numpy as np
from datasets import load_dataset


def load_imdb_data(seed: int = 42) -> tuple:
    """
    Load the IMDB 50k dataset (Maas et al., 2011).

    Returns:
        X_train_text: list[str]  - 25,000 training review strings
        X_test_text:  list[str]  - 25,000 test review strings
        y_train:      np.ndarray - binary labels (0=negative, 1=positive)
        y_test:       np.ndarray - binary labels (0=negative, 1=positive)
    """
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    X_train_text = dataset["train"]["text"]
    y_train = np.array(dataset["train"]["label"])

    X_test_text = dataset["test"]["text"]
    y_test = np.array(dataset["test"]["label"])

    print(f"  Training samples: {len(X_train_text)}")
    print(f"  Test samples:     {len(X_test_text)}")
    print(f"  Label distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Label distribution (test):  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train_text, X_test_text, y_train, y_test


def load_imdb_subset(n_train: int = 5000, n_test: int = 2000, seed: int = 42) -> tuple:
    """
    Load a smaller subset for quick experimentation / debugging.
    Maintains class balance via stratified sampling.
    """
    X_train_text, X_test_text, y_train, y_test = load_imdb_data(seed=seed)

    rng = np.random.RandomState(seed)

    # Stratified subsample for train
    train_idx = _stratified_sample(y_train, n_train, rng)
    test_idx = _stratified_sample(y_test, n_test, rng)

    X_train_sub = [X_train_text[i] for i in train_idx]
    X_test_sub = [X_test_text[i] for i in test_idx]

    print(f"\n  Subset — Train: {len(X_train_sub)}, Test: {len(X_test_sub)}")
    return X_train_sub, X_test_sub, y_train[train_idx], y_test[test_idx]


def _stratified_sample(labels: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Return indices for a stratified subsample of size n."""
    classes = np.unique(labels)
    per_class = n // len(classes)
    indices = []
    for c in classes:
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, size=min(per_class, len(class_idx)), replace=False)
        indices.append(chosen)
    return np.concatenate(indices)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_imdb_data()
    print(f"\nSample review (first 200 chars):\n{X_train[0][:200]}...")
    print(f"Label: {'positive' if y_train[0] == 1 else 'negative'}")
