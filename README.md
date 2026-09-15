# CardioRisk — Heart Disease Risk Prediction

A clinical decision-support demo that predicts a patient's risk of heart
disease from their vitals, trained on the UCI Cleveland Heart Disease
dataset. Built with scikit-learn and Streamlit.

**Live demo:** https://heart-disease-prediction-quick.streamlit.app

## What it does

- Predicts heart disease risk from 13 clinical features (age, chest pain
  type, cholesterol, resting ECG, etc.) using a Random Forest classifier.
- Shows a live risk gauge and the top features driving each prediction.
- Includes a dataset explorer (with the original CSV available for
  download) and a model insights tab with SHAP explainability.

## Dataset

[UCI Machine Learning Repository — Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease),
303 patients, 14 attributes. The original multi-class target (0 = no
disease, 1–4 = increasing severity) is collapsed into a binary label
(0 = no disease, 1 = disease present). After dropping rows with missing
values, 297 records remain. The original CSV is included in this repo at
`data/heart_disease_uci_original.csv`.

## Model performance

Random Forest classifier, evaluated on a held-out 20% stratified test split:

| Metric   | Score |
|----------|-------|
| Accuracy | 86.7% |
| AUC      | 0.941 |

The strongest predictors (by both feature importance and SHAP) are chest
pain type (`cp`), thalassemia status (`thal`), and maximum heart rate
achieved (`thalach`) — consistent with established cardiology literature.

## Project structure

```
├── app.py                              # Streamlit app (3 tabs: predict, dataset, model insights)
├── train_model.py                      # Reproducible training script
├── data/
│   ├── heart_disease_uci_original.csv  # Original dataset, as sourced from UCI
│   └── heart_disease_processed.csv     # Cleaned + binary-labeled version used for training
├── model/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── feature_importances.csv
├── assets/
│   └── shap_summary.png                # Precomputed SHAP summary plot
├── .streamlit/config.toml              # App theme
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model from scratch:

```bash
python train_model.py
```

## Limitations

This is a portfolio/demo project, not a certified clinical tool. The
dataset is small (297 usable records) and comes from a single hospital
cohort, so performance may not generalize to other populations without
further validation.

## Tech stack

Python, scikit-learn, pandas, Streamlit, Plotly, SHAP
