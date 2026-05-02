import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

_explainer_cache = {}

def get_shap_explainer(model, X_train):
    if "shap" not in _explainer_cache:
        explainer = shap.TreeExplainer(model)
        _explainer_cache["shap"] = explainer
    return _explainer_cache["shap"]

def get_lime_explainer(X_train, feature_names):
    if "lime" not in _explainer_cache:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=["No Churn", "Churn"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        _explainer_cache["lime"] = explainer
    return _explainer_cache["lime"]

def _predict_fn(model, feature_names):
    """Wraps predict_proba to accept numpy arrays without feature name warnings."""
    def predict(X_array):
        df = pd.DataFrame(X_array, columns=feature_names)
        return model.predict_proba(df)
    return predict

# ── SHAP: global feature importance ──────────────────────────────────────────

def shap_global_importance(model, X_sample, feature_names, top_n=15):
    explainer = get_shap_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif np.array(shap_values).ndim == 3:
        sv = np.array(shap_values)[:, :, 1]
        sv = sv.T if sv.shape[0] != X_sample.shape[0] else sv
    else:
        sv = shap_values

    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)

    df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": mean_abs.flatten().tolist()
    }).sort_values("importance", ascending=False).head(top_n)

    return df

# ── SHAP: single-customer waterfall ──────────────────────────────────────────

def shap_waterfall_fig(model, X_train, customer_row, feature_names):
    explainer = get_shap_explainer(model, X_train)
    shap_values = explainer(customer_row)

    if shap_values.values.ndim == 3:
        sv = shap.Explanation(
            values=shap_values.values[0, :, 1],
            base_values=shap_values.base_values[0, 1],
            data=shap_values.data[0],
            feature_names=list(feature_names)
        )
    else:
        sv = shap_values[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(sv, max_display=12, show=False)
    plt.tight_layout()
    return plt.gcf()

# ── SHAP: beeswarm summary plot ───────────────────────────────────────────────

def shap_beeswarm_fig(model, X_sample, feature_names, top_n=15):
    explainer = get_shap_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif np.array(shap_values).ndim == 3:
        sv = np.array(shap_values)[:, :, 1]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=list(feature_names),
        max_display=top_n,
        show=False,
        plot_size=None
    )
    plt.tight_layout()
    return plt.gcf()

# ── LIME: single-customer explanation ────────────────────────────────────────

def lime_explanation(model, X_train, customer_row, feature_names, top_n=10):
    lime_exp = get_lime_explainer(X_train, feature_names)

    exp = lime_exp.explain_instance(
        data_row=customer_row.values[0],
        predict_fn=_predict_fn(model, list(feature_names)),
        num_features=top_n,
        top_labels=2
    )

    available_labels = list(exp.local_exp.keys())
    label = 1 if 1 in available_labels else available_labels[-1]

    fig = exp.as_pyplot_figure(label=label)
    plt.tight_layout()

    raw = exp.as_list(label=label)
    explanation_df = pd.DataFrame(raw, columns=["condition", "weight"])
    explanation_df["direction"] = explanation_df["weight"].apply(
        lambda w: "increases_churn" if w > 0 else "decreases_churn"
    )
    explanation_df["abs_weight"] = explanation_df["weight"].abs()
    explanation_df = explanation_df.sort_values("abs_weight", ascending=False)

    return fig, explanation_df

# ── Business recommendations from LIME output ────────────────────────────────

def generate_recommendations(explanation_df, churn_probability):
    recommendations = []
    risk_drivers = explanation_df[
        explanation_df["direction"] == "increases_churn"
    ].head(5)

    FEATURE_ADVICE = {
        "Contract":         "Offer a discounted 1- or 2-year contract to reduce churn risk.",
        "tenure":           "Long-tenure customers are loyal — consider a loyalty reward.",
        "MonthlyCharges":   "High monthly charges are a churn signal — review the pricing plan.",
        "InternetService":  "Fiber optic customers churn more — check service quality or pricing.",
        "TechSupport":      "Customers without tech support churn more — offer a free trial.",
        "OnlineSecurity":   "No online security correlates with churn — bundle it at a discount.",
        "PaymentMethod":    "Electronic check payers churn more — incentivise auto-pay.",
        "PaperlessBilling": "Paperless billing users churn more — ensure clear billing UX.",
        "Dependents":       "No dependents means less lock-in — highlight family plan value.",
        "Partner":          "Solo customers churn more — consider referral incentives.",
    }

    for _, row in risk_drivers.iterrows():
        condition = row["condition"]
        matched = False
        for key, advice in FEATURE_ADVICE.items():
            if key.lower() in condition.lower():
                recommendations.append({
                    "feature":   key,
                    "condition": condition,
                    "advice":    advice,
                    "weight":    round(row["abs_weight"], 4)
                })
                matched = True
                break
        if not matched:
            recommendations.append({
                "feature":   condition.split(" ")[0],
                "condition": condition,
                "advice":    f"Review customer profile regarding: {condition}",
                "weight":    round(row["abs_weight"], 4)
            })

    if churn_probability >= 0.7:
        level, color = "HIGH RISK", "red"
    elif churn_probability >= 0.4:
        level, color = "MEDIUM RISK", "orange"
    else:
        level, color = "LOW RISK", "green"

    return recommendations, level, color