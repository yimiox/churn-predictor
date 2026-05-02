import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
from src.data_loader import load_and_preprocess, get_train_test
from src.model import load_model, evaluate_model
from src.explainer import (
    shap_global_importance, shap_waterfall_fig,
    shap_beeswarm_fig, lime_explanation, generate_recommendations
)

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1E2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2E3250;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.85rem; color: #9BA3C2; margin: 0; }
    .risk-high   { color: #FF4B6E; }
    .risk-medium { color: #FFB347; }
    .risk-low    { color: #4BFF9F; }
    .rec-card {
        background: #1E2130;
        border-left: 4px solid #6C63FF;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #9BA3C2;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data & model loading (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_everything():
    X_train, X_test, y_train, y_test, features = get_train_test()
    model   = load_model()
    metrics = evaluate_model(model, X_test, y_test)
    scaler  = joblib.load("models/scaler.pkl")
    return model, X_train, X_test, y_train, y_test, features, metrics, scaler

model, X_train, X_test, y_train, y_test, features, metrics, scaler = load_everything()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Predict Single Customer", "📦 Bulk Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** Random Forest")
    st.markdown(f"**AUC Score:** `{metrics['auc']}`")
    st.markdown(f"**Training rows:** `{len(X_train)}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Customer Churn Dashboard")
    st.markdown("Model performance, global feature drivers, and dataset insights.")
    st.markdown("---")

    # ── KPI row ──────────────────────────────────────────────────────────────
    report = metrics["report"]
    churn_rate = round(y_test.mean() * 100, 1)
    precision  = round(report["1"]["precision"] * 100, 1)
    recall     = round(report["1"]["recall"] * 100, 1)
    f1         = round(report["1"]["f1-score"] * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, color in [
        (c1, "Test AUC",       metrics["auc"],   "#6C63FF"),
        (c2, "Churn Rate",     f"{churn_rate}%", "#FF4B6E"),
        (c3, "Precision",      f"{precision}%",  "#4BFF9F"),
        (c4, "Recall",         f"{recall}%",     "#FFB347"),
        (c5, "F1 Score",       f"{f1}%",         "#38BDF8"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value" style="color:{color}">{value}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: ROC curve + Confusion matrix ──────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">ROC Curve</p>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"],
            mode="lines", name=f"AUC = {metrics['auc']}",
            line=dict(color="#6C63FF", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#9BA3C2", dash="dash", width=1),
            name="Random"
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="#1E2130", paper_bgcolor="#0F1117",
            font=dict(color="#FAFAFA"),
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(bgcolor="#1E2130")
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = np.array(metrics["cm"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Churn", "Churn"],
            y=["No Churn", "Churn"],
            color_continuous_scale=[[0, "#1E2130"], [1, "#6C63FF"]]
        )
        fig_cm.update_layout(
            plot_bgcolor="#1E2130", paper_bgcolor="#0F1117",
            font=dict(color="#FAFAFA"),
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Row 3: Global SHAP importance ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Global Feature Importance (SHAP)</p>', unsafe_allow_html=True)

    with st.spinner("Computing SHAP values..."):
        importance_df = shap_global_importance(
            model, X_train.sample(300, random_state=42), features
        )

    fig_imp = px.bar(
        importance_df.sort_values("importance"),
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#2E3250", "#6C63FF"],
        labels={"importance": "Mean |SHAP|", "feature": ""}
    )
    fig_imp.update_layout(
        plot_bgcolor="#1E2130", paper_bgcolor="#0F1117",
        font=dict(color="#FAFAFA"),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=10, b=40),
        height=450
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Row 4: SHAP beeswarm ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">SHAP Beeswarm — Feature Impact Direction</p>', unsafe_allow_html=True)
    st.caption("Red = high feature value · Blue = low · Horizontal position = impact on churn probability")

    with st.spinner("Generating beeswarm plot..."):
        fig_bee = shap_beeswarm_fig(
            model, X_train.sample(300, random_state=42), features
        )
    st.pyplot(fig_bee, use_container_width=True)
    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT SINGLE CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Single Customer":
    st.title("Single Customer Prediction")
    st.markdown("Fill in customer details to get a churn prediction with SHAP + LIME explanations.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown('<p class="section-header">Customer Profile</p>', unsafe_allow_html=True)

        with st.form("customer_form"):
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
            total_charges   = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                               value=float(tenure * monthly_charges))

            gender          = st.selectbox("Gender", ["Male", "Female"])
            senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner         = st.selectbox("Partner", ["No", "Yes"])
            dependents      = st.selectbox("Dependents", ["No", "Yes"])
            phone_service   = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec      = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protect  = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless       = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment         = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    with col_result:
        if submitted:
            # Build raw input dict matching original CSV columns
            raw = {
                "tenure": tenure, "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner, "Dependents": dependents,
                "PhoneService": phone_service, "MultipleLines": multiple_lines,
                "InternetService": internet, "OnlineSecurity": online_sec,
                "OnlineBackup": online_backup, "DeviceProtection": device_protect,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment
            }

            # Preprocess to match training schema
            full_df = load_and_preprocess()
            template = full_df.drop("Churn", axis=1).iloc[0:0].copy()

            input_df = pd.DataFrame([raw])
            input_df["TotalCharges"] = pd.to_numeric(
                input_df["TotalCharges"], errors="coerce"
            ).fillna(0)

            # Encode + align with training columns
            from sklearn.preprocessing import LabelEncoder
            binary_cols = [c for c in input_df.columns
                           if input_df[c].nunique() <= 2 and input_df[c].dtype == object]
            le = LabelEncoder()
            for col in binary_cols:
                input_df[col] = le.fit_transform(input_df[col].astype(str))

            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=template.columns, fill_value=0)

            # Scale numeric
            num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            # Predict
            proba = model.predict_proba(input_df)[0][1]
            pred  = int(proba >= 0.5)

            # ── Risk badge ───────────────────────────────────────────────────
            if proba >= 0.7:
                risk_class, risk_label, gauge_color = "risk-high", "HIGH RISK", "#FF4B6E"
            elif proba >= 0.4:
                risk_class, risk_label, gauge_color = "risk-medium", "MEDIUM RISK", "#FFB347"
            else:
                risk_class, risk_label, gauge_color = "risk-low", "LOW RISK", "#4BFF9F"

            st.markdown(f"""
            <div class="metric-card" style="text-align:center; margin-bottom:1rem">
                <p class="metric-label">Churn Probability</p>
                <p class="metric-value {risk_class}">{round(proba*100,1)}%</p>
                <p style="color:{gauge_color}; font-weight:600">{risk_label}</p>
            </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={"suffix": "%", "font": {"color": gauge_color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#9BA3C2"},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "#1E2130",
                    "steps": [
                        {"range": [0, 40],   "color": "#12351F"},
                        {"range": [40, 70],  "color": "#3A2E0F"},
                        {"range": [70, 100], "color": "#3A0F1A"},
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.75, "value": round(proba * 100, 1)
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0F1117", font=dict(color="#FAFAFA"),
                margin=dict(l=30, r=30, t=20, b=10), height=200
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── SHAP waterfall ───────────────────────────────────────────────
            st.markdown('<p class="section-header">SHAP — Why this prediction?</p>',
                        unsafe_allow_html=True)
            with st.spinner("Running SHAP..."):
                fig_wf = shap_waterfall_fig(model, X_train, input_df, features)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close("all")

            # ── LIME explanation ─────────────────────────────────────────────
            st.markdown('<p class="section-header">LIME — Local explanation</p>',
                        unsafe_allow_html=True)
            with st.spinner("Running LIME..."):
                fig_lime, exp_df = lime_explanation(model, X_train, input_df, features)
            st.pyplot(fig_lime, use_container_width=True)
            plt.close("all")

            # ── Recommendations ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<p class="section-header">Business Recommendations</p>',
                        unsafe_allow_html=True)
            recs, level, _ = generate_recommendations(exp_df, proba)

            if recs:
                for r in recs:
                    st.markdown(f"""
                    <div class="rec-card">
                        <strong>{r['feature']}</strong>
                        <span style="color:#9BA3C2; font-size:0.8rem"> · impact {r['weight']}</span><br>
                        <span style="font-size:0.85rem; color:#CBD0E8">{r['condition']}</span><br>
                        <span style="color:#A78BFA">→ {r['advice']}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No major churn drivers detected for this customer.")

        else:
            st.info("👈 Fill in the customer profile and click **Predict**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BULK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Bulk Analysis":
    st.title("Bulk Customer Analysis")
    st.markdown("Upload a CSV of customers to get churn predictions for the entire file.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV (same format as training data)", type="csv")

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw_df)} customers loaded**")

        with st.spinner("Preprocessing and predicting..."):
            # Preprocess
            df = raw_df.copy()
            if "customerID" in df.columns:
                ids = df["customerID"]
                df.drop("customerID", axis=1, inplace=True)
            else:
                ids = pd.Series(range(len(df)), name="customerID")

            if "Churn" in df.columns:
                df.drop("Churn", axis=1, inplace=True)

            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

            from sklearn.preprocessing import LabelEncoder
            binary_cols = [c for c in df.columns
                           if df[c].nunique() == 2 and df[c].dtype == object]
            le = LabelEncoder()
            for col in binary_cols:
                df[col] = le.fit_transform(df[col].astype(str))

            df = pd.get_dummies(df)
            full_df  = load_and_preprocess()
            template = full_df.drop("Churn", axis=1).iloc[0:0]
            df = df.reindex(columns=template.columns, fill_value=0)

            num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            df[num_cols] = scaler.transform(df[num_cols])

            probas = model.predict_proba(df)[:, 1]
            preds  = (probas >= 0.5).astype(int)

        results = raw_df.copy()
        results["Churn_Probability"] = (probas * 100).round(1)
        results["Prediction"]        = np.where(preds == 1, "Will Churn", "Will Stay")
        results["Risk_Level"]        = pd.cut(
            probas,
            bins=[0, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        )

        # ── Summary KPIs ─────────────────────────────────────────────────────
        n_high   = (results["Risk_Level"] == "High").sum()
        n_medium = (results["Risk_Level"] == "Medium").sum()
        n_low    = (results["Risk_Level"] == "Low").sum()
        avg_prob = round(probas.mean() * 100, 1)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "High Risk",    n_high,   "#FF4B6E"),
            (c2, "Medium Risk",  n_medium, "#FFB347"),
            (c3, "Low Risk",     n_low,    "#4BFF9F"),
            (c4, "Avg Churn %",  f"{avg_prob}%", "#6C63FF"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">{label}</p>
                <p class="metric-value" style="color:{color}">{val}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk distribution chart ───────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<p class="section-header">Risk Distribution</p>',
                        unsafe_allow_html=True)
            risk_counts = results["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_dist = px.pie(
                risk_counts, names="Risk Level", values="Count",
                color="Risk Level",
                color_discrete_map={"High": "#FF4B6E", "Medium": "#FFB347", "Low": "#4BFF9F"},
                hole=0.45
            )
            fig_dist.update_layout(
                paper_bgcolor="#0F1117", font=dict(color="#FAFAFA"),
                legend=dict(bgcolor="#1E2130"),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_r:
            st.markdown('<p class="section-header">Probability Distribution</p>',
                        unsafe_allow_html=True)
            fig_hist = px.histogram(
                results, x="Churn_Probability", nbins=30,
                color_discrete_sequence=["#6C63FF"]
            )
            fig_hist.update_layout(
                plot_bgcolor="#1E2130", paper_bgcolor="#0F1117",
                font=dict(color="#FAFAFA"),
                xaxis_title="Churn Probability (%)",
                yaxis_title="Customers",
                margin=dict(l=20, r=20, t=10, b=40)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Sortable results table ────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">Full Results</p>', unsafe_allow_html=True)

        show_cols = ["Churn_Probability", "Prediction", "Risk_Level",
                     "tenure", "MonthlyCharges", "Contract"]
        if "customerID" in results.columns:
            show_cols = ["customerID"] + show_cols

        available = [c for c in show_cols if c in results.columns]
        display_df = results[available].sort_values(
            "Churn_Probability", ascending=False
        ).reset_index(drop=True)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=420,
            column_config={
                "Churn_Probability": st.column_config.ProgressColumn(
                    "Churn %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Risk_Level": st.column_config.TextColumn("Risk"),
            }
        )

        # ── Download button ───────────────────────────────────────────────────
        csv_out = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download full results CSV",
            data=csv_out,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.info("👆 Upload a CSV file to get started. Use the original Telco dataset format.")
        st.markdown("**Expected columns include:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, etc.")