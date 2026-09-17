# SHAP vs. LIME: An Empirical Comparison of Post-Hoc Explainability Methods

To go to the app directly, click **[Click here](https://farismutapcic-shap-lime-project-shap-lime-projectapp-ska8py.streamlit.app/)**.

An empirical comparison of two popular post-hoc explainability methods, **SHAP** and **LIME**, using the Breast Cancer Wisconsin dataset. The project investigates whether the methods agree on important features, how stable their explanations are under small input changes, their computational cost, and whether the findings generalize across different machine-learning models.

**Key result:** On 50 Random Forest test instances, SHAP and LIME achieved a mean Top-5 Jaccard similarity of **0.676**, while the simulated random baseline was **0.101**. SHAP also showed higher explanation stability than LIME under input perturbations.

## Contents

* [Results](#results)
* [The data](#the-data)
* [Models](#models)
* [Experiments](#experiments)
* [Findings](#findings)
* [Repository layout](#repository-layout)
* [The app](#the-app)
* [Reproducing](#reproducing)
* [Limitations](#limitations)

---

## Results

The main experiments compare SHAP and LIME explanations for the same predictions.

### Experiment A — Feature Importance Agreement

50 random test instances were analyzed using the Top-5 features from SHAP and LIME. Agreement was measured using **Jaccard similarity**.

| Model               | Mean Jaccard | Std. Dev. | 95% Bootstrap CI |
| ------------------- | -----------: | --------: | ---------------: |
| Random Forest       |    **0.676** |     0.312 |   [0.590, 0.765] |
| Logistic Regression |    **0.749** |     0.248 |   [0.681, 0.816] |

The simulated random Top-5 baseline had a mean Jaccard similarity of **0.101** for both models.

The Random Forest comparison produced a t-statistic of **13.038** and Cohen's *d* of **1.844** against the random baseline.

### Experiment B — Explanation Stability

The explanations were tested after adding small perturbations to the input features.

For the Random Forest at **3% of feature standard deviation**:

| Method | Mean stability | Std. Dev. |
| ------ | -------------: | --------: |
| SHAP   |     **0.9892** |    0.0349 |
| LIME   |     **0.9099** |    0.1362 |

A paired Wilcoxon signed-rank test was used to compare the methods. The experiment also tested several noise levels: **1%, 3%, 5%, and 10%**.

### Experiment C — Computational Cost

The project measures the time required to generate one explanation.

| Method                 | Mean time per explanation |
| ---------------------- | ------------------------: |
| SHAP — TreeExplainer   |            **0.000192 s** |
| SHAP — KernelExplainer |            **0.063679 s** |
| LIME                   |            **0.067960 s** |

TreeExplainer is specifically optimized for tree-based models, while KernelExplainer provides a model-agnostic comparison that is closer to LIME.

---

## The data

The project uses the **Breast Cancer Wisconsin dataset**, available directly through `scikit-learn`.

|                  |                         |
| ---------------- | ----------------------- |
| **Dataset**      | Breast Cancer Wisconsin |
| **Observations** | 569                     |
| **Features**     | 30 numerical features   |
| **Task**         | Binary classification   |
| **Classes**      | Malignant / Benign      |

The dataset is loaded directly using `sklearn.datasets.load_breast_cancer`, so no external dataset download is required.

The data is divided into training and testing sets using an **80/20 split** with stratification and `random_state=42`.

---

## Models

Two different classification models are evaluated.

### Random Forest

The main model uses:

```text
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)
```

The trained Random Forest achieved:

* **Training accuracy:** 0.9868
* **Test accuracy:** 0.9561

### Logistic Regression

A second model is used to test whether the SHAP/LIME findings generalize beyond Random Forest.

The Logistic Regression model achieved a **test accuracy of 0.9649**.

For Logistic Regression, SHAP uses `KernelExplainer` with `predict_proba` so that SHAP and LIME operate on the same probability scale.

---

## Experiments

### Experiment A — Feature Importance Agreement

Tests whether SHAP and LIME identify similar important features for the same prediction.

The Top-5 features from both methods are compared using:

**Jaccard similarity**

```text
J(A,B) = |A ∩ B| / |A ∪ B|
```

The experiment uses 50 randomly selected test instances and compares the observed agreement against a simulated random baseline.

### Experiment B — Perturbation Stability

Tests how much explanations change when small amounts of noise are added to the input.

The experiment evaluates stability using cosine similarity and tests noise levels of:

```text
1% → 3% → 5% → 10%
```

A paired Wilcoxon signed-rank test is used for the main comparison.

### Experiment C — Computational Cost

Measures the time required to generate explanations using:

* SHAP TreeExplainer
* SHAP KernelExplainer
* LIME

Bootstrap 95% confidence intervals are calculated for the measured execution times.

### Experiment D — Clinical Sensibility

The project examines explanations for:

* Confidently correct predictions
* Confidently wrong predictions
* Borderline predictions

The analysis looks at whether important features correspond to recognizable size, shape, and texture measurements.

This is a **qualitative analysis**, not clinical validation. The experiment uses only three examples per group.

### Experiment E — Generalization

Experiments A and B are repeated using Logistic Regression to determine whether the observed behavior is specific to Random Forest or also appears with a structurally different model.

For Logistic Regression:

* Mean Top-5 Jaccard: **0.749**
* SHAP stability at 3% noise: **0.9549**
* LIME stability at 3% noise: **0.8928**

---

## Findings

**1. SHAP and LIME generally agree, but not perfectly.**

The mean Top-5 Jaccard similarity was approximately **0.68 for Random Forest** and **0.75 for Logistic Regression**, showing substantial agreement but also differences between the two explanation methods.

**2. SHAP showed greater stability under perturbations.**

At the 3% noise level, SHAP had higher mean cosine similarity than LIME for both Random Forest and Logistic Regression. The project reports statistically significant paired comparisons.

**3. Computational cost depends on the SHAP explainer.**

TreeExplainer is extremely fast for the Random Forest because it uses the tree structure directly. When comparing the model-agnostic SHAP KernelExplainer with LIME, their measured computational costs are much closer.

**4. The results were tested on two different model types.**

The project evaluates both Random Forest and Logistic Regression rather than relying on a single architecture.

---


## The app

An interactive application is provided for exploring individual predictions and their SHAP and LIME explanations.

The app allows users to:

* Enter the 30 dataset features
* Generate a model prediction
* View the prediction probability
* Generate a SHAP explanation
* Generate a LIME explanation
* Compare SHAP and LIME feature importance
* Explore the project experiments

**To open the deployed application:**

[Open the SHAP vs LIME App](https://farismutapcic-shap-lime-project-shap-lime-projectapp-ska8py.streamlit.app/)


---

## Reproducing

The easiest way to reproduce the results is to run the notebook directly in **Google Colab**. This avoids the need to clone the repository, create a virtual environment, or manually configure Python on your computer.

The notebook can also be run locally in another Jupyter environment if preferred.

### Google Colab

Simply open the notebook in **Google Colab** and run the cells from top to bottom. The required libraries can be installed directly in the notebook.

### Running Locally

If you prefer to run the project locally, clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install shap lime scikit-learn pandas matplotlib scipy joblib
```

The project uses:

```text
Python 3.12
```

For reproducibility, the notebook uses:

```text
random_state = 42
```

---

## Limitations

* The dataset contains **569 observations**, so the study is relatively small.
* Only two model types are tested: Random Forest and Logistic Regression.
* Experiment D is qualitative and is **not clinical validation**.
* Experiment A uses 50 test instances.
* Experiment B uses 15 independent base instances with five perturbations per instance.
* Experiment C uses 50 instances for TreeExplainer/LIME and 25 for KernelExplainer because KernelExplainer is slower
* Results may differ for other datasets, models, feature types, or experimental settings.

---

## Project objective

The overall goal is to empirically investigate the behavior of **SHAP and LIME as post-hoc explanation methods**, focusing on their agreement, stability, computational cost, and behavior across different machine-learning models.

The results provide quantitative evidence that the two methods do not always produce identical explanations and that explanation stability and computational cost depend on the explainer and model being used.
