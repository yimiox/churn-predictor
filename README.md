# Customer Churn Predictor

An end-to-end ML pipeline that predicts customer churn with full explainability.

## Live Demo
https://churn-predictor-e3xywtyfjjud7atrulkggx.streamlit.app/

## Features
- **Random Forest classifier** trained on IBM Telco Churn dataset (AUC ~0.84)
- **SHAP** global feature importance + per-customer waterfall plots
- **LIME** local surrogate explanations with plain-English business recommendations
- **3-page Streamlit dashboard**: Overview · Single Prediction · Bulk CSV Analysis

## Tech Stack
`scikit-learn` · `shap` · `lime` · `streamlit` · `plotly` · `pandas`

## Run Locally
```bash
pip install -r requirements.txt
python train.py          # trains and saves model to models/
streamlit run app.py     # launches the app
```

## Model Performance
| Metric | Score |
|--------|-------|
| AUC    | 0.842 |
| Precision (churn) | 54.4% |
| Recall (churn)    | 74.1% |
| F1 (churn)        | 62.7% |
