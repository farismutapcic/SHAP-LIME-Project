# SHAP-LIME-Project

# # How to Use the SHAP vs LIME Dashboard

## 1. Open the App

Open the Streamlit app in your web browser.

You will start on the **Home** page.

The Home page gives you a quick explanation of the project and what SHAP and LIME do.

---

## 2. Go to Prediction

Click **Prediction** in the menu.

Here you can enter information about a breast cancer dataset example.

You will see 30 boxes where you can enter numbers.

The boxes are grouped into:

* **Mean Features**
* **Error Features**
* **Worst Features**

Enter the required values into the boxes.

Then click **Predict**.

---

## 3. See the Prediction

After clicking **Predict**, the app will show:

* The predicted class
* The prediction probability
* Information about the model

The model used in the project is a **Random Forest** model.

**Important:** This prediction is for research and educational purposes. It is **not a medical diagnosis**.

---

## 4. View the SHAP Explanation

After making a prediction, go to **SHAP**.

SHAP shows you which features had the biggest effect on the prediction.

You can choose to see:

* Top 5 features
* Top 10 features
* Top 15 features

The app displays the features in a chart so you can easily see which ones were more important.

In simple words:

**SHAP helps answer: "Which features helped influence the model's prediction?"**

---

## 5. View the LIME Explanation

Go to **LIME** after making a prediction.

LIME also shows which features are important for the individual prediction.

You can choose:

* Top 5
* Top 10
* Top 15

In simple words:

**LIME helps answer: "Which features were important for this particular prediction?"**

---

## 6. Compare SHAP and LIME

Go to **Comparison**.

Here you can see SHAP and LIME next to each other.

The app compares the important features selected by both methods.

You will also see **Jaccard similarity**.

You do not need to calculate it yourself.

It simply tells you **how much SHAP and LIME agree about the important features**.

For example:

* A higher similarity means the selected features overlap more.
* A lower similarity means the selected features overlap less.

---

## 7. Look at the Experiments

Go to **Experiments**.

This section shows the research experiments from the project.

There are five experiments:

### Experiment A — Top-Feature Agreement

Checks how much SHAP and LIME agree about the most important features.

### Experiment B — Perturbation Stability

Checks how much the explanations change when small changes or noise are added to the input.

### Experiment C — Computational Cost

Compares how much time SHAP and LIME need to generate explanations.

### Experiment D — Qualitative Clinical Sensibility

Looks at whether the explanations appear sensible for correct, wrong, and borderline predictions.

This is a **qualitative** experiment and does not prove medical validity.

### Experiment E — Generalization to a Second Model

Checks whether the comparison between SHAP and LIME is similar when using another machine learning model.

---

## 8. Check Computational Cost

Go to **Computational Cost**.

This page focuses specifically on how long SHAP and LIME take to generate explanations.

You can see the results in charts and tables.

---

## 9. Read About the Project

Go to **About**.

This page explains:

* The research goal
* The dataset
* The machine learning model
* What SHAP is
* What LIME is
* What Jaccard similarity means
* The five experiments
* The limitations of the research

---

# Simple User Journey

If you don't know where to start, just follow these steps:

**Home**

↓

**Prediction** → Enter the 30 values → Click **Predict**

↓

**SHAP** → See what features influenced the prediction

↓

**LIME** → See what features influenced the prediction

↓

**Comparison** → Compare SHAP and LIME

↓

**Experiments** → Explore the research experiments

↓

**Computational Cost** → Compare their execution times

↓

**About** → Learn more about the project

---

## Important

You do **not** need to understand SHAP, LIME, machine learning, or Jaccard similarity to use the basic app.

Just remember:

**Prediction** = What does the model predict?

**SHAP** = Why did the model make this prediction?

**LIME** = What features were important for this individual prediction?

**Comparison** = How similar are SHAP and LIME?

**Experiments** = What did the research test?

**Computational Cost** = How much time do SHAP and LIME take?

**About** = What is the project about?

> **Disclaimer:** This application is for research and educational purposes only. It is not a medical diagnostic tool.
