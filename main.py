"""
main.py
-------
Entry point for the PCA + SVM Sentiment Analysis project.
Runs all experiments and generates all visualizations for the report.

Usage:
    python main.py                  # Full run (all 50k reviews)
    python main.py --subset 5000     # Quick run with subset (for debugging)
"""

import argparse
import os
import time
import numpy as np

# ─── Project modules ─────────────────────────────────────────────────────────
from data_loader import load_imdb_data, load_imdb_subset
from preprocessing import (
    build_tfidf,
    plot_tfidf_sparsity,
    plot_top_tfidf_terms,
    plot_document_length_distribution,
)
from pca_analysis import (
    fit_pca, transform_pca, get_explained_variance,
    find_n_components_for_variance,
    plot_scree, plot_cumulative_variance_zoom,
    plot_2d_projection, plot_3d_projection,
    plot_component_top_words, plot_pairwise_components,
)
from svm_classifier import (
    train_svm, evaluate_svm,
    plot_confusion_matrix, plot_confusion_matrix_comparison,
    plot_decision_boundary_2d, plot_decision_value_distribution,
    plot_svm_top_features,
)
from experiments import (
    run_baseline, run_pca_sweep,
    plot_accuracy_vs_components, plot_f1_vs_components,
    plot_time_vs_components, plot_accuracy_vs_variance,
    plot_summary_table,
)
from visualization import (
    plot_wordclouds, plot_eigenvalue_spectrum,
    plot_component_variance_heatmap, plot_metrics_radar,
    plot_dimensionality_dashboard,
)


def main():
    parser = argparse.ArgumentParser(description="PCA + SVM Sentiment Analysis")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use a subset of N training samples (for quick testing)")
    parser.add_argument("--max-features", type=int, default=10_000,
                        help="Maximum TF-IDF vocabulary size (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    t_start = time.time()

    SEED = args.seed
    np.random.seed(SEED)

    # ─── Step 1: Load Data ───────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 1: Loading IMDB Dataset")
    print("█" * 60)

    if args.subset:
        X_train_text, X_test_text, y_train, y_test = load_imdb_subset(
            n_train=args.subset, n_test=args.subset // 3, seed=SEED
        )
    else:
        X_train_text, X_test_text, y_train, y_test = load_imdb_data(seed=SEED)

    # ─── Step 2: Preprocessing ───────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 2: TF-IDF Vectorization & Data Exploration")
    print("█" * 60)

    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf(
        X_train_text, X_test_text, max_features=args.max_features
    )

    # Data exploration plots
    plot_document_length_distribution(X_train_text, y_train)
    plot_tfidf_sparsity(X_train_tfidf)
    plot_top_tfidf_terms(vectorizer, X_train_tfidf, y_train)
    plot_wordclouds(vectorizer, X_train_tfidf, y_train)

    # ─── Step 3: Baseline SVM ────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 3: Baseline SVM (No Dimensionality Reduction)")
    print("█" * 60)

    baseline = run_baseline(X_train_tfidf, X_test_tfidf, y_train, y_test)

    plot_confusion_matrix(
        baseline["metrics"]["confusion_matrix"],
        title="Baseline SVM Confusion Matrix",
        save_path="results/cm_baseline.png",
    )
    plot_decision_value_distribution(
        baseline["metrics"]["decision_values"], y_test,
        title="Baseline SVM — Decision Value Distribution",
        save_path="results/decision_dist_baseline.png",
    )
    plot_svm_top_features(baseline["model"], vectorizer)
    print(f"\n  Classification Report:\n{baseline['metrics']['report']}")

    # ─── Step 4: PCA Analysis ────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 4: PCA / SVD Analysis")
    print("█" * 60)

    # Fit a large PCA to analyze the variance spectrum
    n_analysis = min(2000, args.max_features - 1)
    pca_full, X_train_pca_full = fit_pca(X_train_tfidf, n_components=n_analysis, seed=SEED)

    # Variance analysis
    thresholds = find_n_components_for_variance(pca_full)
    print("\n  Components needed for variance thresholds:")
    for t, n in thresholds.items():
        print(f"    {t:.0%} variance → {n} components")

    # PCA visualizations
    plot_scree(pca_full)
    plot_cumulative_variance_zoom(pca_full)
    plot_eigenvalue_spectrum(pca_full)
    plot_component_variance_heatmap(pca_full, n_show=20)
    plot_component_top_words(pca_full, vectorizer)

    # 2D & 3D projections (use first 2-3 components from the full model)
    X_train_2d = X_train_pca_full[:, :2]
    X_train_3d = X_train_pca_full[:, :3]
    plot_2d_projection(X_train_2d, y_train)
    plot_3d_projection(X_train_3d, y_train)
    plot_pairwise_components(X_train_pca_full[:, :4], y_train)

    # Decision boundary visualization on 2D
    plot_decision_boundary_2d(X_train_2d, y_train)

    # ─── Step 5: PCA Sweep Experiments ───────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 5: PCA Component Sweep — Core Experiment")
    print("█" * 60)

    component_counts = [5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    sweep_df = run_pca_sweep(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        component_counts=component_counts, seed=SEED,
    )

    # Save raw results
    sweep_df.drop(columns=["confusion_matrix"]).to_csv(
        "results/sweep_results.csv", index=False
    )
    print("  Saved: results/sweep_results.csv")

    # ─── Step 6: Generate All Comparison Plots ───────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 6: Generating Comparison Plots")
    print("█" * 60)

    baseline_acc = baseline["metrics"]["accuracy"]
    baseline_f1 = baseline["metrics"]["f1"]
    baseline_time = baseline["train_time"]

    plot_accuracy_vs_components(sweep_df, baseline_acc)
    plot_f1_vs_components(sweep_df, baseline_f1)
    plot_time_vs_components(sweep_df, baseline_time)
    plot_accuracy_vs_variance(sweep_df, baseline_acc)
    plot_summary_table(sweep_df, baseline)
    plot_dimensionality_dashboard(sweep_df, baseline_acc, baseline_time)

    # Find best PCA config for comparison plots
    best_idx = sweep_df["accuracy"].idxmax()
    best_row = sweep_df.loc[best_idx]
    best_n = int(best_row["n_components"])
    best_cm = best_row["confusion_matrix"]

    plot_confusion_matrix_comparison(
        baseline["metrics"]["confusion_matrix"],
        best_cm,
        n_components=best_n,
    )

    # Radar chart: baseline vs best PCA
    # Re-run best PCA to get full metrics
    pca_best_model, X_train_best = fit_pca(X_train_tfidf, n_components=best_n, seed=SEED)
    X_test_best = transform_pca(pca_best_model, X_test_tfidf)
    svm_best = train_svm(X_train_best, y_train)
    best_metrics = evaluate_svm(svm_best["model"], X_test_best, y_test)

    plot_metrics_radar(baseline["metrics"], best_metrics, n_components=best_n)

    # Decision value distributions for best PCA
    plot_decision_value_distribution(
        best_metrics["decision_values"], y_test,
        title=f"PCA-{best_n} + SVM — Decision Value Distribution",
        save_path="results/decision_dist_pca_best.png",
    )

    # ─── Done ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "█" * 60)
    print(f"  COMPLETE — Total runtime: {elapsed:.1f}s")
    print(f"  All plots saved to: results/")
    print("█" * 60)

    # List all generated files
    print("\n  Generated files:")
    for f in sorted(os.listdir("results")):
        fpath = os.path.join("results", f)
        size = os.path.getsize(fpath) / 1024
        print(f"    {f:40s} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
