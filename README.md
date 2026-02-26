# SVM Semantic Analysis

Use **PCA** (Principal Component Analysis) and **SVM** (Support Vector Machines) to perform **sentiment analysis** on the IMDB 50k movie review dataset. The project explores whether PCA preprocessing improves or degrades SVM classification accuracy on high-dimensional TF-IDF text data, and compares computational cost and visual separability of classes.

For full context, research questions, and bibliography, see **[docs/](docs/)** (e.g. `docs/project_proposal.tex`).

---

## Overview

- **Data:** 50,000 IMDB movie reviews (Maas et al., 2011) — 25k train / 25k test, binary sentiment (positive/negative).
- **Pipeline:** Raw text → TF-IDF vectorization → (optional) PCA dimensionality reduction → Linear SVM classification.
- **Central question:** Does PCA improve SVM accuracy by removing noise, or hurt it by discarding useful information?
- **Output:** Baseline vs PCA sweep experiments, accuracy/F1/time vs number of components, visualizations (scree, 2D/3D projections, confusion matrices, decision boundaries, word clouds, etc.).

---

## Requirements

- **Python 3.9+** (tested with 3.10+)
- Dependencies in `requirements.txt`:
  - `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `datasets` (HuggingFace; used to load IMDB)
  - `wordcloud`

---

## Setup

1. **Clone or download** the project and go to its root:
   ```bash
   cd SVM_semantic_analysis
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional:** The first run will download the IMDB dataset via the `datasets` library (requires internet).

---

## Running the Project

All steps (data load, TF-IDF, baseline SVM, PCA analysis, component sweep, and all plots) are run from a single entry point:

```bash
source .venv/bin/activate  # if not already in venv
python main.py
```

- Uses the **full** IMDB 50k dataset (25k train / 25k test).
- Creates a **`results/`** directory and writes all figures and `sweep_results.csv` there.
- Full run can take several minutes depending on hardware.

### Quick / debug run (subset)

To test with a smaller subset and shorter runtime:

```bash
python main.py --subset 5000
```

- Uses 5,000 training and 1,667 test samples (stratified).
- Same pipeline and outputs, but faster.

### Other options

| Option            | Default   | Description                                      |
|-------------------|-----------|--------------------------------------------------|
| `--subset N`      | None      | Use N training samples (and N/3 test) for speed |
| `--max-features`  | 10,000    | Max TF-IDF vocabulary size                      |
| `--seed`          | 42        | Random seed for reproducibility                 |

Examples:

```bash
python main.py --subset 3000 --max-features 5000
python main.py --seed 0
```

---

## What the Pipeline Does

1. **Load data** — IMDB train/test splits (and optional subset).
2. **Preprocessing** — TF-IDF vectorization; plots for document length, sparsity, top terms, word clouds.
3. **Baseline SVM** — LinearSVC on full TF-IDF (no PCA); confusion matrix, decision-value distribution, top features.
4. **PCA analysis** — Fit PCA (up to 2000 components for analysis); scree plot, cumulative variance, eigenvalue spectrum, component–word associations; 2D/3D projections and decision boundary in 2D.
5. **PCA sweep** — Train SVM at several component counts (e.g. 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000); record accuracy, F1, train time; save `results/sweep_results.csv`.
6. **Comparison plots** — Accuracy/F1/time vs components, accuracy vs variance, summary table, dimensionality dashboard, confusion matrix comparison (baseline vs best PCA), metrics radar, decision-value distribution for best PCA.

All figures are saved under **`results/`**.

---

## Project Structure

| Path                | Purpose                                      |
|---------------------|----------------------------------------------|
| `main.py`           | Entry point; runs full pipeline              |
| `data_loader.py`    | Load IMDB via HuggingFace `datasets`         |
| `preprocessing.py`  | TF-IDF and exploratory text plots            |
| `pca_analysis.py`   | PCA fit/transform, variance, 2D/3D plots     |
| `svm_classifier.py` | LinearSVC train/eval and SVM visualizations  |
| `experiments.py`    | Baseline and PCA-sweep experiments           |
| `visualization.py`  | Extra plots (word clouds, eigenvalue, etc.)  |
| `requirements.txt`  | Python dependencies                          |
| **`docs/`**         | Project proposal and references              |

---

## Documentation

- **`docs/project_proposal.tex`** — Project proposal (topic, central question, secondary questions, annotated bibliography).
- **`docs/references.bib`** — Bibliography (e.g. Maas et al. IMDB, PCA/SVM and dimensionality reduction references).

---

## Citation (dataset)

When using the IMDB data, cite:

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *ACL*.
