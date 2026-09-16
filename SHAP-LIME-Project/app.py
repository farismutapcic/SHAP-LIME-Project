"""
SHAP vs LIME Explainability Dashboard

"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import shap
from lime.lime_tabular import LimeTabularExplainer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

st.set_page_config(
    page_title="SHAP vs LIME Explainability Dashboard",
    page_icon="🔬",
    layout="wide",
)

SHAP_COLOR = "#4C72B0"
LIME_COLOR = "#DD8452"
KERNEL_COLOR = "#55A868"

DISCLAIMER = (
    "This application is for research and educational purposes only. "
    "It is not a medical diagnostic tool."
)

# ----------------------------------------------------------------------------
# Shared utility functions (identical to the notebook's "Shared utility
# functions" section / utils.py)
# ----------------------------------------------------------------------------


def jaccard(set_a, set_b):
    """Jaccard similarity between two iterables of feature names."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(np.dot(a, b) / denom)


def bootstrap_ci(values, n_boot=5000, ci=95, rng=None, seed=RANDOM_STATE):
    """Percentile bootstrap CI for the mean of `values`."""
    rng = rng if rng is not None else np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.randint(0, n, size=n)].mean()
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi


def cohens_d_one_sample(values, reference_mean):
    values = np.asarray(values)
    return (values.mean() - reference_mean) / values.std(ddof=1)


def expected_random_jaccard(n_features, top_k, n_trials=20000, rng=None, seed=RANDOM_STATE):
    """Simulated random baseline: Jaccard similarity between two
    independently-random top-k feature subsets, out of n_features total."""
    rng = rng if rng is not None else np.random.RandomState(seed)
    sims = np.empty(n_trials)
    idx = np.arange(n_features)
    for i in range(n_trials):
        a = set(rng.choice(idx, size=top_k, replace=False))
        b = set(rng.choice(idx, size=top_k, replace=False))
        sims[i] = len(a & b) / len(a | b)
    return sims


def extract_binary_shap_vector(sv, positive_class_index=1):
    """Normalize SHAP's shap_values() output (which varies in shape by SHAP
    version / explainer type) into a flat 1D vector for the positive class,
    for a single instance."""
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if sv.shape[-1] in (2,) and sv.shape[0] == 1:
            return sv[0, :, positive_class_index]
        if sv.shape[0] == 2:
            return sv[positive_class_index, 0, :]
        return sv[0, :, positive_class_index]
    if sv.ndim == 2:
        return sv[0]
    return sv


def holm_bonferroni(p_values, alpha=0.05):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    m = len(p_values)
    corrected = np.empty(m)
    reject = np.empty(m, dtype=bool)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
        reject[idx] = corrected[idx] < alpha
    return corrected, reject


CLINICALLY_CANONICAL_SUBSTRINGS = (
    "radius", "perimeter", "area", "concavity", "concave points", "texture",
)


def _is_canonical(feature_name):
    return any(s in feature_name for s in CLINICALLY_CANONICAL_SUBSTRINGS)


def _canonical_fraction(top_features):
    if not top_features:
        return float("nan")
    return sum(_is_canonical(f) for f in top_features) / len(top_features)


class ProbabilityScaleKernelExplainer:
    """Wraps shap.KernelExplainer with a fixed nsamples so it exposes the
    same .shap_values(X) interface as TreeExplainer."""

    def __init__(self, predict_proba_fn, background, nsamples=200):
        self._explainer = shap.KernelExplainer(predict_proba_fn, background)
        self._nsamples = nsamples

    def shap_values(self, X):
        return self._explainer.shap_values(X, nsamples=self._nsamples, silent=True)


# ----------------------------------------------------------------------------
# Data / model (cached)
# ----------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Mean Features": [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
    ],
    "Error Features": [
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
    ],
    "Worst Features": [
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension",
    ],
}


@st.cache_resource(show_spinner="Loading data and training model...")
def load_and_train():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    shap_explainer = shap.TreeExplainer(model)
    lime_explainer = LimeTabularExplainer(
        X_train.values, feature_names=feature_names,
        class_names=["malignant", "benign"], mode="classification",
        random_state=RANDOM_STATE,
    )
    return {
        "X": X, "y": y, "feature_names": feature_names,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "model": model, "train_acc": train_acc, "test_acc": test_acc,
        "shap_explainer": shap_explainer, "lime_explainer": lime_explainer,
        "target_names": list(data.target_names),
    }


CTX = load_and_train()
FEATURE_NAMES = CTX["feature_names"]


def get_shap_top_features(explainer, instance_df, top_k):
    sv = explainer.shap_values(instance_df.values)
    vals = extract_binary_shap_vector(sv)
    order = np.argsort(-np.abs(vals))[:top_k]
    return [FEATURE_NAMES[i] for i in order], vals


def get_lime_top_features(lime_explainer, model, instance_row, top_k):
    exp = lime_explainer.explain_instance(
        instance_row.values, model.predict_proba, num_features=top_k
    )
    label = list(exp.local_exp.keys())[0]
    pairs = exp.local_exp[label][:top_k]
    top_feats = [FEATURE_NAMES[idx] for idx, _w in pairs]
    weight_map = {FEATURE_NAMES[idx]: w for idx, w in exp.local_exp[label]}
    return top_feats, weight_map


def full_shap_vector(explainer, instance_values):
    sv = explainer.shap_values(instance_values.reshape(1, -1))
    return extract_binary_shap_vector(sv)


def full_lime_vector(lime_explainer, model, instance_values):
    exp = lime_explainer.explain_instance(
        instance_values, model.predict_proba, num_features=len(FEATURE_NAMES)
    )
    label = list(exp.local_exp.keys())[0]
    weight_map = {idx: w for idx, w in exp.local_exp[label]}
    return np.array([weight_map.get(i, 0.0) for i in range(len(FEATURE_NAMES))])


# ----------------------------------------------------------------------------
# Cached experiment runners (adapted 1:1 from the notebook's standalone scripts)
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner="Running Experiment A (SHAP/LIME feature agreement)...")
def run_experiment_a(_model_tag="RandomForest", n_samples=50, top_k=5):
    model = CTX["model"]
    X_test = CTX["X_test"]
    shap_explainer = CTX["shap_explainer"]
    lime_explainer = CTX["lime_explainer"]

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=n_samples, replace=False)
    samples = X_test.iloc[sample_idx].reset_index(drop=True)

    results = []
    for i in range(n_samples):
        instance_df = samples.iloc[[i]]
        instance_row = samples.iloc[i]
        shap_top, _ = get_shap_top_features(shap_explainer, instance_df, top_k)
        lime_top, _ = get_lime_top_features(lime_explainer, model, instance_row, top_k)
        j = jaccard(shap_top, lime_top)
        results.append({
            "instance": i, "shap_top_features": shap_top,
            "lime_top_features": lime_top, "jaccard_similarity": j,
        })
    df = pd.DataFrame(results)

    mean_jaccard = df["jaccard_similarity"].mean()
    ci_lo, ci_hi = bootstrap_ci(df["jaccard_similarity"].values)
    random_baseline = expected_random_jaccard(len(FEATURE_NAMES), top_k)
    t_stat, p_value = stats.ttest_1samp(df["jaccard_similarity"], random_baseline.mean())
    d = cohens_d_one_sample(df["jaccard_similarity"], random_baseline.mean())

    stats_summary = {
        "n_samples": n_samples, "top_k": top_k,
        "mean_jaccard": mean_jaccard, "std_jaccard": df["jaccard_similarity"].std(),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "random_baseline_mean": random_baseline.mean(),
        "t_stat": t_stat, "p_value": p_value, "cohens_d": d,
    }
    return df, stats_summary


def run_zero_noise_control(model, X_test, lime_explainer, shap_explainer, sample_idx):
    records = []
    for i, idx in enumerate(sample_idx):
        base_values = X_test.iloc[idx].values.astype(float)
        shap_a = full_shap_vector(shap_explainer, base_values)
        shap_b = full_shap_vector(shap_explainer, base_values)
        lime_a = full_lime_vector(lime_explainer, model, base_values)
        lime_b = full_lime_vector(lime_explainer, model, base_values)
        records.append({
            "instance": i,
            "shap_self_similarity": cosine_sim(shap_a, shap_b),
            "lime_self_similarity": cosine_sim(lime_a, lime_b),
        })
    return pd.DataFrame(records)


def run_stability_at_noise_level(model, X_test, feature_stds, shap_explainer,
                                  lime_explainer, noise_level, rng, n_instances=15,
                                  n_perturb=5, sample_idx=None):
    if sample_idx is None:
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_test), size=n_instances, replace=False
        )
    records = []
    for i, idx in enumerate(sample_idx):
        base_values = X_test.iloc[idx].values.astype(float)
        base_shap = full_shap_vector(shap_explainer, base_values)
        base_lime = full_lime_vector(lime_explainer, model, base_values)
        for p in range(n_perturb):
            noise = rng.normal(0, noise_level * feature_stds)
            perturbed = base_values + noise
            pert_shap = full_shap_vector(shap_explainer, perturbed)
            pert_lime = full_lime_vector(lime_explainer, model, perturbed)
            records.append({
                "instance": i, "perturbation": p, "noise_level": noise_level,
                "shap_cosine_similarity": cosine_sim(base_shap, pert_shap),
                "lime_cosine_similarity": cosine_sim(base_lime, pert_lime),
            })
    return pd.DataFrame(records)


@st.cache_data(show_spinner="Running Experiment B (stability under perturbation)... this can take a minute")
def run_experiment_b(_model_tag="RandomForest", n_instances=15, n_perturb=5,
                      noise_level=0.03, sweep=(0.01, 0.03, 0.05, 0.10)):
    model = CTX["model"]
    X_test = CTX["X_test"]
    X_train = CTX["X_train"]
    shap_explainer = CTX["shap_explainer"]
    lime_explainer = CTX["lime_explainer"]
    feature_stds = X_train.std().values

    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_test), size=n_instances, replace=False
    )
    control_df = run_zero_noise_control(model, X_test, lime_explainer, shap_explainer, sample_idx)

    rng = np.random.RandomState(RANDOM_STATE)
    df = run_stability_at_noise_level(
        model, X_test, feature_stds, shap_explainer, lime_explainer, noise_level,
        rng, n_instances=n_instances, n_perturb=n_perturb, sample_idx=sample_idx,
    )

    instance_means = df.groupby("instance")[["shap_cosine_similarity", "lime_cosine_similarity"]].mean()
    w_stat, p_value = stats.wilcoxon(
        instance_means["shap_cosine_similarity"], instance_means["lime_cosine_similarity"]
    )

    sweep_records = [df]
    for nl in sweep:
        if nl == noise_level:
            continue
        rng_level = np.random.RandomState(RANDOM_STATE + int(nl * 1000))
        df_level = run_stability_at_noise_level(
            model, X_test, feature_stds, shap_explainer, lime_explainer, nl,
            rng_level, n_instances=n_instances, n_perturb=n_perturb, sample_idx=sample_idx,
        )
        sweep_records.append(df_level)
    sweep_df = pd.concat(sweep_records, ignore_index=True)

    summary = {
        "shap_self_mean": control_df["shap_self_similarity"].mean(),
        "lime_self_mean": control_df["lime_self_similarity"].mean(),
        "shap_mean": df["shap_cosine_similarity"].mean(),
        "lime_mean": df["lime_cosine_similarity"].mean(),
        "w_stat": w_stat, "p_value": p_value,
        "n_instances": n_instances, "noise_level": noise_level,
    }
    return df, sweep_df, control_df, summary


@st.cache_data(show_spinner="Running Experiment C (computational cost)...")
def run_experiment_c(n_instances=50, n_kernel_instances=25, kernel_nsamples=500):
    model = CTX["model"]
    X_train = CTX["X_train"]
    X_test = CTX["X_test"]
    lime_explainer = CTX["lime_explainer"]

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=n_instances, replace=False)
    samples = X_test.iloc[sample_idx].values

    tree_explainer = shap.TreeExplainer(model)
    tree_times = []
    for row in samples:
        t0 = time.perf_counter()
        tree_explainer.shap_values(row.reshape(1, -1))
        tree_times.append(time.perf_counter() - t0)

    lime_times = []
    for row in samples:
        t0 = time.perf_counter()
        lime_explainer.explain_instance(row, model.predict_proba, num_features=len(FEATURE_NAMES))
        lime_times.append(time.perf_counter() - t0)

    background = shap.sample(X_train, 50, random_state=RANDOM_STATE)
    kernel_explainer = shap.KernelExplainer(model.predict_proba, background)
    kernel_times = []
    for row in samples[:n_kernel_instances]:
        t0 = time.perf_counter()
        kernel_explainer.shap_values(row.reshape(1, -1), nsamples=kernel_nsamples, silent=True)
        kernel_times.append(time.perf_counter() - t0)

    results = pd.DataFrame({
        "method": (["SHAP (TreeExplainer)"] * len(tree_times)
                   + ["LIME"] * len(lime_times)
                   + ["SHAP (KernelExplainer)"] * len(kernel_times)),
        "time_seconds": tree_times + lime_times + kernel_times,
    })

    summary = results.groupby("method")["time_seconds"].agg(["mean", "std", "count"])
    ci_rows = []
    for method, group in results.groupby("method"):
        lo, hi = bootstrap_ci(group["time_seconds"].values)
        ci_rows.append({"method": method, "ci_lo": lo, "ci_hi": hi})
    ci_df = pd.DataFrame(ci_rows).set_index("method")
    summary = summary.join(ci_df).sort_values("mean")
    return results, summary


@st.cache_data(show_spinner="Running Experiment D (qualitative clinical sensibility)...")
def run_experiment_d(top_k=5):
    model = CTX["model"]
    X_test = CTX["X_test"]
    y_test = CTX["y_test"]
    shap_explainer = CTX["shap_explainer"]
    lime_explainer = CTX["lime_explainer"]

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    confidence = probs.max(axis=1)
    correct_mask = preds == y_test.values

    confident_correct_idx = np.argsort(-confidence * correct_mask)[:3]
    wrong_idx_all = np.where(~correct_mask)[0]
    confident_wrong_idx = (
        wrong_idx_all[np.argsort(-confidence[wrong_idx_all])][:3] if len(wrong_idx_all) > 0 else []
    )
    borderline_idx = np.argsort(np.abs(confidence - 0.5))[:3]

    group_records = {}
    example_rows = []

    def describe(idx_list, group_key, group_label):
        rows = []
        for idx in idx_list:
            row = X_test.iloc[idx].values.astype(float)
            true_label = y_test.iloc[idx]
            pred_label = preds[idx]
            conf = confidence[idx]

            sv = extract_binary_shap_vector(shap_explainer.shap_values(row.reshape(1, -1)))
            shap_top = [FEATURE_NAMES[i] for i in np.argsort(-np.abs(sv))[:top_k]]

            exp = lime_explainer.explain_instance(row, model.predict_proba, num_features=top_k)
            lime_label = list(exp.local_exp.keys())[0]
            lime_top = [FEATURE_NAMES[fidx] for fidx, _w in exp.local_exp[lime_label][:top_k]]

            shap_frac = _canonical_fraction(shap_top)
            lime_frac = _canonical_fraction(lime_top)
            rows.append({"idx": idx, "shap_canonical_fraction": shap_frac, "lime_canonical_fraction": lime_frac})
            example_rows.append({
                "group": group_label, "idx": idx, "true_class": CTX["target_names"][true_label],
                "pred_class": CTX["target_names"][pred_label], "confidence": conf,
                "shap_top_features": shap_top, "lime_top_features": lime_top,
            })
        group_records[group_key] = pd.DataFrame(rows)

    describe(confident_correct_idx, "correct", "Confidently correct")
    if len(confident_wrong_idx) > 0:
        describe(confident_wrong_idx, "wrong", "Confidently wrong")
    else:
        group_records["wrong"] = pd.DataFrame(columns=["idx", "shap_canonical_fraction", "lime_canonical_fraction"])
    describe(borderline_idx, "borderline", "Borderline")

    summary_rows = []
    for key, label in [("correct", "Confidently correct"), ("wrong", "Confidently wrong"), ("borderline", "Borderline")]:
        gdf = group_records[key]
        shap_mean = gdf["shap_canonical_fraction"].mean() if len(gdf) else float("nan")
        lime_mean = gdf["lime_canonical_fraction"].mean() if len(gdf) else float("nan")
        summary_rows.append({"group": label, "shap_canonical_fraction": shap_mean, "lime_canonical_fraction": lime_mean})
    summary_df = pd.DataFrame(summary_rows)
    examples_df = pd.DataFrame(example_rows)
    return summary_df, examples_df


@st.cache_data(show_spinner="Running Experiment E (generalization to Logistic Regression)... this can take a minute or two")
def run_experiment_e(n_samples_a=50, top_k=5, n_instances_b=15, n_perturb=5, noise_level=0.03):
    X_train = CTX["X_train"]
    X_test = CTX["X_test"]
    y_train = CTX["y_train"]
    y_test = CTX["y_test"]
    model = CTX["model"]

    lr_model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
    rf_test_acc = accuracy_score(y_test, model.predict(X_test))

    background = shap.sample(X_train, 50, random_state=RANDOM_STATE)
    lr_shap_explainer = ProbabilityScaleKernelExplainer(lr_model.predict_proba, background)
    lr_lime_explainer = LimeTabularExplainer(
        X_train.values, feature_names=FEATURE_NAMES,
        class_names=["malignant", "benign"], mode="classification",
        random_state=RANDOM_STATE,
    )

    # Experiment A for LR
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=n_samples_a, replace=False)
    samples = X_test.iloc[sample_idx].reset_index(drop=True)
    results_a = []
    for i in range(n_samples_a):
        instance_df = samples.iloc[[i]]
        instance_row = samples.iloc[i]
        shap_top, _ = get_shap_top_features(lr_shap_explainer, instance_df, top_k)
        lime_top, _ = get_lime_top_features(lr_lime_explainer, lr_model, instance_row, top_k)
        results_a.append({"instance": i, "jaccard_similarity": jaccard(shap_top, lime_top)})
    df_a_lr = pd.DataFrame(results_a)

    # Experiment B for LR
    feature_stds = X_train.std().values
    rng_b = np.random.RandomState(RANDOM_STATE)
    df_b_lr = run_stability_at_noise_level(
        lr_model, X_test, feature_stds, lr_shap_explainer, lr_lime_explainer,
        noise_level, rng_b, n_instances=n_instances_b, n_perturb=n_perturb,
    )

    df_a_rf, _ = run_experiment_a(n_samples=n_samples_a, top_k=top_k)
    df_b_rf, _, _, _ = run_experiment_b(n_instances=n_instances_b, n_perturb=n_perturb, noise_level=noise_level)

    model_comparison = pd.DataFrame({
        "model": ["RandomForest", "LogisticRegression"],
        "test_accuracy": [rf_test_acc, lr_test_acc],
        "mean_jaccard": [df_a_rf["jaccard_similarity"].mean(), df_a_lr["jaccard_similarity"].mean()],
        "shap_stability": [df_b_rf["shap_cosine_similarity"].mean(), df_b_lr["shap_cosine_similarity"].mean()],
        "lime_stability": [df_b_rf["lime_cosine_similarity"].mean(), df_b_lr["lime_cosine_similarity"].mean()],
    })
    return model_comparison, df_a_lr, df_b_lr


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

def init_state():
    defaults = {
        "features": {f: float(CTX["X"][f].mean()) for f in FEATURE_NAMES},
        "prediction": None, "prediction_proba": None,
        "shap_top": None, "shap_vals": None,
        "lime_top": None, "lime_weights": None,
        "page": "Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------

st.sidebar.title("🔬 SHAP vs LIME")
st.sidebar.caption("Explainability Dashboard")
PAGES = ["Home", "Prediction", "SHAP Explanation", "LIME Explanation",
         "Comparison", "Experiments", "Computational Cost", "About"]
page = st.sidebar.radio("Navigate", PAGES, index=PAGES.index(st.session_state["page"]))
st.session_state["page"] = page

st.sidebar.divider()
if st.session_state["prediction"] is not None:
    st.sidebar.success(
        f"Last prediction: **{CTX['target_names'][st.session_state['prediction']]}** "
        f"(p={st.session_state['prediction_proba']:.3f})"
    )
else:
    st.sidebar.info("No prediction yet — go to **Prediction** to get started.")
st.sidebar.divider()
st.sidebar.caption(f"Model test accuracy: {CTX['test_acc']:.3f}")
st.sidebar.caption(DISCLAIMER)

# ============================================================================
# HOME
# ============================================================================
if page == "Home":
    st.title("SHAP vs LIME Explainability Dashboard")
    st.markdown(
        "This project empirically compares **SHAP** and **LIME** as post-hoc "
        "explainability methods for a machine learning model trained on the "
        "**Breast Cancer Wisconsin** dataset."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is SHAP?")
        st.write(
            "**SHapley Additive exPlanations** describe how individual features "
            "contribute to a model prediction, based on cooperative game theory."
        )
    with c2:
        st.subheader("What is LIME?")
        st.write(
            "**Local Interpretable Model-agnostic Explanations** explains an "
            "individual prediction by approximating the model locally with an "
            "interpretable model."
        )

    st.divider()
    st.subheader("Get started")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🎯 Prediction")
        st.write("Enter the 30 tumor features and run the model.")
        if st.button("Go to Prediction", use_container_width=True):
            st.session_state["page"] = "Prediction"
            st.rerun()
    with b:
        st.markdown("#### 🔵 SHAP Explanation")
        st.write("See which features drove the prediction, SHAP's way.")
        if st.button("Go to SHAP", use_container_width=True):
            st.session_state["page"] = "SHAP Explanation"
            st.rerun()
    with c:
        st.markdown("#### 🟠 LIME Explanation")
        st.write("See which features drove the prediction, LIME's way.")
        if st.button("Go to LIME", use_container_width=True):
            st.session_state["page"] = "LIME Explanation"
            st.rerun()

    st.divider()
    d, e = st.columns(2)
    with d:
        st.markdown("#### ⚖️ Comparison")
        st.write("Compare SHAP vs LIME top features with Jaccard similarity.")
        if st.button("Go to Comparison", use_container_width=True):
            st.session_state["page"] = "Comparison"
            st.rerun()
    with e:
        st.markdown("#### 🧪 Experiments")
        st.write("Explore the actual research experiments (A–E).")
        if st.button("Go to Experiments", use_container_width=True):
            st.session_state["page"] = "Experiments"
            st.rerun()

    st.divider()
    st.subheader("About the Project")
    st.write(
        "The research trains a depth-constrained `RandomForestClassifier` on the "
        "Breast Cancer Wisconsin dataset and runs five experiments (A–E) comparing "
        "SHAP and LIME on feature agreement, stability under perturbation, "
        "computational cost, qualitative clinical sensibility, and generalization "
        "to a second model (Logistic Regression). All numbers on the Experiments "
        "page are computed live by this app, not hard-coded."
    )
    st.warning(DISCLAIMER)

# ============================================================================
# PREDICTION
# ============================================================================
elif page == "Prediction":
    st.title("🎯 Prediction")
    st.warning(DISCLAIMER)
    st.write(
        "Enter the 30 Breast Cancer Wisconsin features below (grouped into "
        "Mean / Error / Worst), then click **Predict**."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Reset to dataset mean values"):
            for f in FEATURE_NAMES:
                st.session_state["features"][f] = float(CTX["X"][f].mean())
            st.rerun()
    with c2:
        if st.button("Load a random test-set example"):
            row = CTX["X_test"].sample(1, random_state=np.random.randint(0, 100000)).iloc[0]
            for f in FEATURE_NAMES:
                st.session_state["features"][f] = float(row[f])
            st.rerun()

    for group_name, feats in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Mean Features")):
            cols = st.columns(2)
            for i, f in enumerate(feats):
                col = cols[i % 2]
                col_min, col_max = float(CTX["X"][f].min()), float(CTX["X"][f].max())
                st.session_state["features"][f] = col.number_input(
                    f, min_value=0.0, max_value=col_max * 2 if col_max > 0 else 1.0,
                    value=float(st.session_state["features"][f]), step=(col_max - col_min) / 100 or 0.01,
                    key=f"input_{f}",
                )

    st.divider()
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        row = pd.DataFrame([st.session_state["features"]])[FEATURE_NAMES]
        model = CTX["model"]
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][pred])
        st.session_state["prediction"] = pred
        st.session_state["prediction_proba"] = proba
        # Clear stale explanations from a previous instance
        st.session_state["shap_top"] = None
        st.session_state["lime_top"] = None
        st.rerun()

    if st.session_state["prediction"] is not None:
        st.divider()
        pred = st.session_state["prediction"]
        proba = st.session_state["prediction_proba"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted class", CTX["target_names"][pred])
        c2.metric("Probability", f"{proba:.3f}")
        c3.metric("Model", "RandomForest (n=100, depth=5)")

        st.markdown("**Model information:** `RandomForestClassifier(n_estimators=100, "
                    "max_depth=5, min_samples_leaf=3, random_state=42)` — "
                    f"test accuracy: {CTX['test_acc']:.3f}")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Generate SHAP explanation →", use_container_width=True):
                st.session_state["page"] = "SHAP Explanation"
                st.rerun()
        with b2:
            if st.button("Generate LIME explanation →", use_container_width=True):
                st.session_state["page"] = "LIME Explanation"
                st.rerun()

# ============================================================================
# SHAP EXPLANATION
# ============================================================================
elif page == "SHAP Explanation":
    st.title("🔵 SHAP Explanation")
    st.info("SHAP values describe how individual features contribute to a model prediction.")

    if st.session_state["prediction"] is None:
        st.warning("Run a prediction first on the **Prediction** page.")
    else:
        top_k = st.radio("Show top:", [5, 10, 15], horizontal=True, index=1)
        row = pd.DataFrame([st.session_state["features"]])[FEATURE_NAMES]
        top_feats, vals = get_shap_top_features(CTX["shap_explainer"], row, top_k)

        order = np.argsort(-np.abs(vals))[:top_k]
        plot_df = pd.DataFrame({
            "feature": [FEATURE_NAMES[i] for i in order],
            "shap_value": [vals[i] for i in order],
            "feature_value": [st.session_state["features"][FEATURE_NAMES[i]] for i in order],
        }).sort_values("shap_value")

        fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.4)))
        colors = [SHAP_COLOR if v >= 0 else "#B0C4DE" for v in plot_df["shap_value"]]
        ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact toward predicted class)")
        ax.set_title(f"Top {top_k} SHAP Feature Importances")
        st.pyplot(fig, use_container_width=True)

        st.dataframe(
            plot_df.rename(columns={
                "feature": "Feature", "shap_value": "SHAP value", "feature_value": "Feature value",
            }).assign(Direction=lambda d: np.where(d["SHAP value"] >= 0, "→ toward predicted class", "→ away from predicted class"))
              .sort_values("SHAP value", ascending=False)[["Feature", "SHAP value", "Feature value", "Direction"]],
            use_container_width=True, hide_index=True,
        )
        st.session_state["shap_top"] = top_feats

# ============================================================================
# LIME EXPLANATION
# ============================================================================
elif page == "LIME Explanation":
    st.title("🟠 LIME Explanation")
    st.info("LIME explains an individual prediction by approximating the model "
            "locally with an interpretable model.")

    if st.session_state["prediction"] is None:
        st.warning("Run a prediction first on the **Prediction** page.")
    else:
        top_k = st.radio("Show top:", [5, 10, 15], horizontal=True, index=1, key="lime_topk")
        row = pd.DataFrame([st.session_state["features"]])[FEATURE_NAMES]
        instance_row = row.iloc[0]
        top_feats, weight_map = get_lime_top_features(CTX["lime_explainer"], CTX["model"], instance_row, top_k)

        plot_df = pd.DataFrame({
            "feature": top_feats,
            "lime_value": [weight_map[f] for f in top_feats],
            "feature_value": [st.session_state["features"][f] for f in top_feats],
        }).sort_values("lime_value")

        fig, ax = plt.subplots(figsize=(8, max(3, top_k * 0.4)))
        colors = [LIME_COLOR if v >= 0 else "#F4C7A1" for v in plot_df["lime_value"]]
        ax.barh(plot_df["feature"], plot_df["lime_value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME importance (local linear weight)")
        ax.set_title(f"Top {top_k} LIME Feature Importances")
        st.pyplot(fig, use_container_width=True)

        st.dataframe(
            plot_df.rename(columns={
                "feature": "Feature", "lime_value": "LIME importance", "feature_value": "Feature value",
            }).assign(Direction=lambda d: np.where(d["LIME importance"] >= 0, "→ toward predicted class", "→ away from predicted class"))
              .sort_values("LIME importance", ascending=False)[["Feature", "LIME importance", "Feature value", "Direction"]],
            use_container_width=True, hide_index=True,
        )
        st.session_state["lime_top"] = top_feats

# ============================================================================
# COMPARISON
# ============================================================================
elif page == "Comparison":
    st.title("⚖️ SHAP vs LIME Comparison")
    st.info("Jaccard similarity measures how much the selected top features overlap between SHAP and LIME.")

    if st.session_state["prediction"] is None:
        st.warning("Run a prediction first on the **Prediction** page.")
    else:
        top_k = st.radio("Top-K features to compare:", [5, 10, 15], horizontal=True, index=0, key="cmp_topk")
        row = pd.DataFrame([st.session_state["features"]])[FEATURE_NAMES]
        instance_row = row.iloc[0]

        shap_top, shap_vals = get_shap_top_features(CTX["shap_explainer"], row, top_k)
        lime_top, lime_weights = get_lime_top_features(CTX["lime_explainer"], CTX["model"], instance_row, top_k)

        j = jaccard(shap_top, lime_top)
        common = sorted(set(shap_top) & set(lime_top))
        shap_only = sorted(set(shap_top) - set(lime_top))
        lime_only = sorted(set(lime_top) - set(shap_top))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jaccard similarity", f"{j:.3f}")
        m2.metric("Common features", len(common))
        m3.metric("SHAP-only", len(shap_only))
        m4.metric("LIME-only", len(lime_only))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**SHAP top-{top_k}**")
            st.write(shap_top)
        with c2:
            st.markdown(f"**LIME top-{top_k}**")
            st.write(lime_top)

        st.divider()
        st.markdown(f"**Common features:** {', '.join(common) if common else '—'}")
        st.markdown(f"**SHAP-only features:** {', '.join(shap_only) if shap_only else '—'}")
        st.markdown(f"**LIME-only features:** {', '.join(lime_only) if lime_only else '—'}")

        st.divider()
        st.subheader("Comparison table")
        all_feats = sorted(set(shap_top) | set(lime_top))
        shap_rank = {f: i + 1 for i, f in enumerate(shap_top)}
        lime_rank = {f: i + 1 for i, f in enumerate(lime_top)}
        shap_val_map = {FEATURE_NAMES[i]: v for i, v in enumerate(shap_vals)}
        table = pd.DataFrame({
            "Feature": all_feats,
            "SHAP Rank": [shap_rank.get(f, "—") for f in all_feats],
            "LIME Rank": [lime_rank.get(f, "—") for f in all_feats],
            "SHAP Value": [round(shap_val_map.get(f, np.nan), 4) for f in all_feats],
            "LIME Value": [round(lime_weights.get(f, np.nan), 4) for f in all_feats],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.divider()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["SHAP ∩ LIME", "SHAP only", "LIME only"],
               [len(common), len(shap_only), len(lime_only)],
               color=["#8172B2", SHAP_COLOR, LIME_COLOR])
        ax.set_ylabel("Number of features")
        ax.set_title(f"Top-{top_k} Feature Set Overlap (Jaccard = {j:.3f})")
        st.pyplot(fig, use_container_width=True)

# ============================================================================
# EXPERIMENTS
# ============================================================================
elif page == "Experiments":
    st.title("🧪 Experiments")
    st.write(
        "These are the actual research experiments from the project notebook. "
        "Results are computed live below (and cached) — nothing here is invented."
    )

    tabs = st.tabs([
        "A — Feature Agreement", "B — Perturbation Stability", "C — Computational Cost",
        "D — Clinical Sensibility", "E — Generalization", "Cost vs Stability",
    ])

    # ---- Experiment A ----
    with tabs[0]:
        st.subheader("Experiment A — Top-Feature Agreement")
        st.write(
            "Tests whether SHAP and LIME agree on which features matter most for the "
            "*same* predictions, measured via Jaccard similarity on top-5 features "
            "across 50 random test instances, compared against a random-selection baseline."
        )
        if st.button("Run / refresh Experiment A"):
            run_experiment_a.clear()
        df_a, stats_a = run_experiment_a()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test instances", stats_a["n_samples"])
        c2.metric("Mean Jaccard", f"{stats_a['mean_jaccard']:.3f}")
        c3.metric("Std dev", f"{stats_a['std_jaccard']:.3f}")
        c4.metric("95% bootstrap CI", f"[{stats_a['ci_lo']:.3f}, {stats_a['ci_hi']:.3f}]")

        c5, c6, c7 = st.columns(3)
        c5.metric("Random baseline", f"{stats_a['random_baseline_mean']:.3f}")
        c6.metric("t-statistic", f"{stats_a['t_stat']:.3f}")
        c7.metric("Cohen's d", f"{stats_a['cohens_d']:.3f}")
        st.caption(f"p-value: {stats_a['p_value']:.3g}")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(df_a["instance"], df_a["jaccard_similarity"], color=SHAP_COLOR)
        ax.axhline(stats_a["mean_jaccard"], color="red", linestyle="--",
                   label=f"Mean = {stats_a['mean_jaccard']:.2f}")
        ax.axhline(stats_a["random_baseline_mean"], color="gray", linestyle=":",
                   label=f"Random baseline = {stats_a['random_baseline_mean']:.2f}")
        ax.set_xlabel("Test instance")
        ax.set_ylabel(f"Jaccard similarity (top-{stats_a['top_k']})")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # ---- Experiment B ----
    with tabs[1]:
        st.subheader("Experiment B — Perturbation Stability")
        st.write(
            "Adds small Gaussian noise to test instances and measures how much each "
            "method's explanation changes (cosine similarity), including a zero-noise "
            "self-similarity control for LIME's own sampling variance, plus a "
            "noise-level sweep."
        )
        if st.button("Run / refresh Experiment B"):
            run_experiment_b.clear()
        df_b, sweep_b, control_b, summary_b = run_experiment_b()

        c1, c2 = st.columns(2)
        c1.metric("SHAP self-similarity (zero-noise)", f"{summary_b['shap_self_mean']:.4f}")
        c2.metric("LIME self-similarity (zero-noise)", f"{summary_b['lime_self_mean']:.4f}")

        c3, c4, c5 = st.columns(3)
        c3.metric("SHAP mean stability", f"{summary_b['shap_mean']:.4f}")
        c4.metric("LIME mean stability", f"{summary_b['lime_mean']:.4f}")
        c5.metric("Wilcoxon p-value (instance-level)", f"{summary_b['p_value']:.3g}")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.boxplot([df_b["shap_cosine_similarity"], df_b["lime_cosine_similarity"]],
                    tick_labels=["SHAP", "LIME"])
        ax.set_ylabel("Cosine similarity to unperturbed explanation")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Stability at noise={summary_b['noise_level']:.2f} (n={summary_b['n_instances']})")
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Noise-level sweep**")
        sweep_summary = sweep_b.groupby("noise_level").agg(
            shap_mean=("shap_cosine_similarity", "mean"), shap_std=("shap_cosine_similarity", "std"),
            lime_mean=("lime_cosine_similarity", "mean"), lime_std=("lime_cosine_similarity", "std"),
        ).reset_index()
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.errorbar(sweep_summary["noise_level"], sweep_summary["shap_mean"],
                     yerr=sweep_summary["shap_std"], marker="o", capsize=4, label="SHAP", color=SHAP_COLOR)
        ax2.errorbar(sweep_summary["noise_level"], sweep_summary["lime_mean"],
                     yerr=sweep_summary["lime_std"], marker="s", capsize=4, label="LIME", color=LIME_COLOR)
        ax2.axhline(summary_b["lime_self_mean"], color="gray", linestyle="--", linewidth=1,
                    label=f"LIME zero-noise floor ({summary_b['lime_self_mean']:.3f})")
        ax2.set_xlabel("Perturbation noise level (fraction of feature std dev)")
        ax2.set_ylabel("Mean cosine similarity")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

    # ---- Experiment C ----
    with tabs[2]:
        st.subheader("Experiment C — Computational Cost")
        st.write("Times how long each method takes to generate a single explanation.")
        if st.button("Run / refresh Experiment C"):
            run_experiment_c.clear()
        results_c, summary_c = run_experiment_c()
        st.dataframe(summary_c, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        means = summary_c["mean"]
        lower_err = means - summary_c["ci_lo"]
        upper_err = summary_c["ci_hi"] - means
        color_map = {"SHAP (TreeExplainer)": SHAP_COLOR, "LIME": LIME_COLOR, "SHAP (KernelExplainer)": KERNEL_COLOR}
        bar_colors = [color_map[m] for m in means.index]
        ax.bar(means.index, means.values, yerr=[lower_err.values, upper_err.values], capsize=5, color=bar_colors)
        ax.set_yscale("log")
        ax.set_ylabel("Mean time per explanation, seconds (log scale)")
        ax.set_title("Computational Cost (error bars = 95% bootstrap CI)")
        st.pyplot(fig, use_container_width=True)

    # ---- Experiment D ----
    with tabs[3]:
        st.subheader("Experiment D — Qualitative Clinical Sensibility")
        st.caption("Qualitative only — does not establish clinical validity.")
        if st.button("Run / refresh Experiment D"):
            run_experiment_d.clear()
        summary_d, examples_d = run_experiment_d()
        st.dataframe(summary_d, use_container_width=True, hide_index=True)
        st.markdown("**Individual examples inspected**")
        st.dataframe(examples_d, use_container_width=True, hide_index=True)
        st.caption(
            "n=3 per group — a manual/automated sanity check, not a statistically "
            "powered or clinician-verified metric."
        )

    # ---- Experiment E ----
    with tabs[4]:
        st.subheader("Experiment E — Generalization to a Second Model")
        st.write("Repeats Experiments A and B on a Logistic Regression model.")
        if st.button("Run / refresh Experiment E (slower)"):
            run_experiment_e.clear()
        model_comparison, df_a_lr, df_b_lr = run_experiment_e()
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(model_comparison["model"], model_comparison["mean_jaccard"], color=[SHAP_COLOR, LIME_COLOR])
        axes[0].set_ylabel("Mean Jaccard similarity (top-5)")
        axes[0].set_title("Experiment A: Agreement by Model")
        axes[0].set_ylim(0, 1)

        width = 0.35
        x = np.arange(len(model_comparison))
        axes[1].bar(x - width / 2, model_comparison["shap_stability"], width, label="SHAP", color=SHAP_COLOR)
        axes[1].bar(x + width / 2, model_comparison["lime_stability"], width, label="LIME", color=LIME_COLOR)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_comparison["model"])
        axes[1].set_ylabel("Mean cosine similarity under perturbation")
        axes[1].set_title("Experiment B: Stability by Model")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        st.pyplot(fig, use_container_width=True)

    # ---- Cost vs Stability ----
    with tabs[5]:
        st.subheader("Combined Cost vs Stability")
        st.write(
            "Puts Experiment B (stability) and Experiment C (cost) on one plot. "
            "Computing SHAP KernelExplainer's stability independently can take a while."
        )
        if st.button("Compute combined view (slow)"):
            model = CTX["model"]
            X_train, X_test = CTX["X_train"], CTX["X_test"]
            feature_stds = X_train.std().values
            _, _, _, summary_b_local = run_experiment_b()
            results_c_local, _ = run_experiment_c()

            background = shap.sample(X_train, 50, random_state=RANDOM_STATE)
            kernel_explainer = ProbabilityScaleKernelExplainer(model.predict_proba, background)
            lime_expl = CTX["lime_explainer"]
            rng_k = np.random.RandomState(RANDOM_STATE)
            df_k = run_stability_at_noise_level(model, X_test, feature_stds, kernel_explainer,
                                                 lime_expl, 0.03, rng_k)
            kernel_stability = df_k["shap_cosine_similarity"].mean()

            timing_summary = results_c_local.groupby("method")["time_seconds"].mean()
            points = {
                "SHAP (TreeExplainer)": {"cost": timing_summary["SHAP (TreeExplainer)"], "stability": summary_b_local["shap_mean"], "color": SHAP_COLOR},
                "SHAP (KernelExplainer)": {"cost": timing_summary["SHAP (KernelExplainer)"], "stability": kernel_stability, "color": KERNEL_COLOR},
                "LIME": {"cost": timing_summary["LIME"], "stability": summary_b_local["lime_mean"], "color": LIME_COLOR},
            }
            fig, ax = plt.subplots(figsize=(7, 5.5))
            for lbl, p in points.items():
                ax.scatter(p["cost"], p["stability"], s=220, color=p["color"], zorder=3, edgecolor="black", linewidth=0.8)
                ax.annotate(lbl, (p["cost"], p["stability"]), textcoords="offset points", xytext=(10, 8), fontsize=9)
            ax.set_xscale("log")
            ax.set_xlabel("Mean cost per explanation, seconds (log scale)")
            ax.set_ylabel("Mean stability under perturbation (3% noise)")
            ax.set_title("Cost vs. Stability Trade-off")
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Click the button above to compute this view — it re-runs a KernelExplainer stability measurement.")

# ============================================================================
# COMPUTATIONAL COST
# ============================================================================
elif page == "Computational Cost":
    st.title("⏱️ Computational Cost")
    st.write("Dedicated view of Experiment C — comparing explanation generation time across methods.")

    if st.button("Run / refresh timing"):
        run_experiment_c.clear()
    results_c, summary_c = run_experiment_c()

    st.dataframe(summary_c.reset_index().rename(columns={
        "method": "Method", "mean": "Mean (s)", "std": "Std (s)", "count": "N explanations",
        "ci_lo": "95% CI low", "ci_hi": "95% CI high",
    }), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = summary_c["mean"]
    lower_err = means - summary_c["ci_lo"]
    upper_err = summary_c["ci_hi"] - means
    color_map = {"SHAP (TreeExplainer)": SHAP_COLOR, "LIME": LIME_COLOR, "SHAP (KernelExplainer)": KERNEL_COLOR}
    bar_colors = [color_map[m] for m in means.index]
    ax.bar(means.index, means.values, yerr=[lower_err.values, upper_err.values], capsize=5, color=bar_colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean time per explanation, seconds (log scale)")
    ax.set_title("Computational Cost of Explanation Methods")
    st.pyplot(fig, use_container_width=True)

    st.caption(
        "SHAP TreeExplainer is fast because it exploits tree structure directly and "
        "is only available for tree-based models. SHAP KernelExplainer is the fairer, "
        "model-agnostic comparison point to LIME."
    )

# ============================================================================
# ABOUT
# ============================================================================
elif page == "About":
    st.title("ℹ️ About This Project")

    st.subheader("Research Objective")
    st.write(
        "The project empirically compares SHAP and LIME as post-hoc explainability "
        "methods and evaluates feature agreement, explanation stability, computational "
        "cost, qualitative sensibility, and generalization across models."
    )

    st.subheader("Dataset")
    st.markdown("- Breast Cancer Wisconsin dataset\n- 569 observations\n- 30 numerical features\n- Binary classification")

    st.subheader("Model")
    st.code(
        "RandomForestClassifier(\n"
        "    n_estimators=100,\n"
        "    max_depth=5,\n"
        "    min_samples_leaf=3,\n"
        "    random_state=42,\n"
        ")"
    )
    st.write(f"Test accuracy on this run: **{CTX['test_acc']:.3f}**")

    st.subheader("SHAP")
    st.write("SHapley Additive exPlanations — provides feature contributions to predictions.")

    st.subheader("LIME")
    st.write("Local Interpretable Model-agnostic Explanations — builds a local approximation around an individual prediction.")

    st.subheader("Jaccard Similarity")
    st.latex(r"J(A,B) = \frac{|A \cap B|}{|A \cup B|}")
    st.write("Used to compare how much SHAP's and LIME's top-feature sets overlap.")

    st.subheader("Experimental Design")
    st.markdown(
        "- **Experiment A** — Top-feature agreement (Jaccard) vs. random baseline\n"
        "- **Experiment B** — Perturbation stability, with a zero-noise self-similarity control and a noise-level sweep\n"
        "- **Experiment C** — Computational cost (TreeExplainer, KernelExplainer, LIME)\n"
        "- **Experiment D** — Qualitative clinical sensibility on correct / wrong / borderline predictions\n"
        "- **Experiment E** — Generalization to a second model (Logistic Regression)"
    )

    st.subheader("Limitations")
    st.markdown(
        "- Explanations are post-hoc\n"
        "- Results depend on the model and dataset\n"
        "- LIME uses local approximations\n"
        "- SHAP and LIME may produce different explanations\n"
        "- The dataset is not a substitute for real clinical deployment\n"
        "- Qualitative sensibility does not establish clinical validity"
    )

    st.warning(DISCLAIMER)
