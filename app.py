"""
Credit Scoring Prediction Dashboard
====================================
A modern, beginner-friendly Streamlit web application for predicting
loan applicant creditworthiness (Good Credit vs. Bad Credit).
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Optional import of training module for self-healing/retraining
try:
    from train_model import train_and_evaluate_model
except ImportError:
    train_and_evaluate_model = None

# ---------------------------------------------------------
# Page Configuration & Theming
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring & Risk Assessment Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern financial UI styling
st.markdown(
    """
    <style>
    /* Metric Card styling */
    .metric-box {
        background: linear-gradient(135deg, rgba(240, 244, 255, 0.8), rgba(248, 250, 255, 0.9));
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-box-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #4b5563;
        letter-spacing: 0.05em;
    }
    .metric-box-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 4px;
    }
    
    /* Result Cards */
    .decision-approved {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        color: #065f46;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .decision-flagged {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        color: #991b1b;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    
    /* Subtle Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0e7ff;
        color: #3730a3;
        margin-right: 6px;
    }
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper Functions: Loading Artifacts & Data
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the trained machine learning pipeline from disk."""
    model_path = os.path.join("model", "credit_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_metrics_and_features():
    """Loads saved evaluation metrics and feature importances."""
    metrics_path = os.path.join("model", "metrics.json")
    features_path = os.path.join("model", "feature_importance.json")

    metrics, features = None, None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            features = json.load(f)
    return metrics, features


@st.cache_data
def load_dataset():
    """Loads the synthetic credit dataset."""
    data_path = "credit_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


# ---------------------------------------------------------
# Model Self-Healing / Auto-Generation
# ---------------------------------------------------------
model = load_trained_model()
metrics, features = load_metrics_and_features()
df = load_dataset()

if model is None or df is None:
    st.warning("⚠️ Model or dataset not detected. Training initial model now...")
    if train_and_evaluate_model is not None:
        with st.spinner("Generating synthetic data and training Random Forest pipeline..."):
            train_and_evaluate_model()
            st.cache_resource.clear()
            st.cache_data.clear()
            model = load_trained_model()
            metrics, features = load_metrics_and_features()
            df = load_dataset()
            st.success("✅ Model and dataset successfully initialized!")
    else:
        st.error("Please run `python train_model.py` in your terminal to train the model first.")
        st.stop()


# ---------------------------------------------------------
# Sidebar Controls & Presets
# ---------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/bank-card-back-side.png",
    width=70,
)
st.sidebar.title("Credit Risk AI")
st.sidebar.caption("Machine Learning Credit Decision Engine")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Applicant Presets")
st.sidebar.caption("Fill applicant form instantly with typical risk profiles:")

preset_selection = st.sidebar.selectbox(
    "Choose a Preset Profile:",
    [
        "Custom Input",
        "🌟 Prime (High Income, High FICO)",
        "⚖️ Moderate (Average Income, Fair FICO)",
        "⚠️ Subprime (Low Income, History of Default)",
    ],
    index=0,
)

# Preset Values mapping
if preset_selection == "🌟 Prime (High Income, High FICO)":
    default_age = 42
    default_income = 115000
    default_home = "MORTGAGE"
    default_emp = 12.0
    default_intent = "HOMEIMPROVEMENT"
    default_loan = 15000
    default_interest = 7.5
    default_default = "N"
    default_score = 780
    default_history = 16
elif preset_selection == "⚖️ Moderate (Average Income, Fair FICO)":
    default_age = 31
    default_income = 48000
    default_home = "RENT"
    default_emp = 4.0
    default_intent = "PERSONAL"
    default_loan = 9000
    default_interest = 13.5
    default_default = "N"
    default_score = 640
    default_history = 6
elif preset_selection == "⚠️ Subprime (Low Income, History of Default)":
    default_age = 25
    default_income = 24000
    default_home = "RENT"
    default_emp = 1.0
    default_intent = "DEBTCONSOLIDATION"
    default_loan = 14000
    default_interest = 21.0
    default_default = "Y"
    default_score = 520
    default_history = 2
else:
    default_age = 33
    default_income = 62000
    default_home = "RENT"
    default_emp = 5.0
    default_intent = "EDUCATION"
    default_loan = 12000
    default_interest = 11.2
    default_default = "N"
    default_score = 690
    default_history = 7

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")
st.sidebar.write(f"• **Algorithm**: Random Forest (150 trees)")
st.sidebar.write(f"• **Dataset Samples**: {len(df) if df is not None else 0:,}")
if metrics:
    st.sidebar.write(f"• **Model Accuracy**: `{metrics['accuracy']*100:.1f}%`")
    st.sidebar.write(f"• **ROC-AUC Score**: `{metrics['roc_auc']:.3f}`")

if st.sidebar.button("🔄 Retrain Model"):
    if train_and_evaluate_model is not None:
        with st.spinner("Retraining model on new synthetic seed..."):
            train_and_evaluate_model()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.sidebar.success("Model retrained successfully!")
            st.rerun()


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("💳 Credit Scoring Prediction System")
st.markdown(
    """
    <div style='margin-bottom: 1.5rem;'>
        <span class='badge'>scikit-learn Pipeline</span>
        <span class='badge'>Random Forest Classifier</span>
        <span class='badge'>Automated Scaling & Encoding</span>
        <span class='badge badge-success'>Live Inference Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Tabs Navigation
tab_pred, tab_perf, tab_data, tab_guide = st.tabs([
    "🎯 Credit Risk Assessor",
    "📊 Model Performance & Metrics",
    "🔍 Dataset Explorer",
    "📘 How It Works & ML Guide",
])


# =========================================================
# TAB 1: Credit Risk Assessor (Interactive Prediction)
# =========================================================
with tab_pred:
    st.subheader("📋 Applicant Information & Loan Details")
    st.caption("Provide applicant financial metrics below to simulate a real-time automated lending assessment.")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Personal & Employment")
            age = st.slider("Applicant Age", min_value=18, max_value=75, value=default_age, help="Age in years.")
            annual_income = st.number_input(
                "Gross Annual Income ($)",
                min_value=10000,
                max_value=300000,
                value=int(default_income),
                step=2500,
                help="Total gross yearly earnings before taxes.",
            )
            employment_years = st.slider(
                "Employment Length (Years)",
                min_value=0.0,
                max_value=40.0,
                value=float(default_emp),
                step=0.5,
                help="Duration at current employment.",
            )
            home_ownership = st.selectbox(
                "Home Ownership Status",
                options=["RENT", "MORTGAGE", "OWN", "OTHER"],
                index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(default_home),
                help="Current residential living arrangement.",
            )

        with col2:
            st.markdown("##### 💰 Loan Application Details")
            loan_amount = st.number_input(
                "Requested Loan Amount ($)",
                min_value=1000,
                max_value=50000,
                value=int(default_loan),
                step=1000,
                help="Principal amount requested for the loan.",
            )
            loan_intent = st.selectbox(
                "Loan Purpose / Intent",
                options=[
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "PERSONAL",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION",
                ],
                index=[
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "PERSONAL",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION",
                ].index(default_intent),
                help="Intended usage of loan funds.",
            )
            interest_rate = st.slider(
                "Proposed Interest Rate (%)",
                min_value=5.0,
                max_value=25.0,
                value=float(default_interest),
                step=0.25,
                help="Annual Percentage Rate (APR).",
            )
            
            # Real-time Loan-to-Income Calculation preview
            lti_ratio = round(loan_amount / annual_income, 3)
            if lti_ratio > 0.35:
                st.warning(f"⚠️ Loan-to-Income: **{lti_ratio*100:.1f}%** (High Debt Burden)")
            else:
                st.info(f"ℹ️ Loan-to-Income: **{lti_ratio*100:.1f}%** (Healthy Ratio)")

        with col3:
            st.markdown("##### 📜 Credit Bureau History")
            credit_score = st.slider(
                "Credit Bureau Score (FICO)",
                min_value=300,
                max_value=850,
                value=int(default_score),
                step=5,
                help="300 to 850 score representing historical repayment behavior.",
            )
            default_on_file = st.radio(
                "Prior Default on Credit File?",
                options=["N", "Y"],
                index=0 if default_default == "N" else 1,
                horizontal=True,
                help="Has the applicant defaulted on any debt in the past?",
            )
            credit_history_years = st.slider(
                "Credit History Length (Years)",
                min_value=1,
                max_value=30,
                value=int(default_history),
                step=1,
                help="Number of years the applicant has held active credit accounts.",
            )

        submit_btn = st.form_submit_button("⚡ Assess Creditworthiness & Predict Risk", use_container_width=True)

    # Prediction Execution
    if submit_btn or "last_prediction" in st.session_state:
        # Prepare input DataFrame for pipeline
        applicant_payload = pd.DataFrame([{
            "age": age,
            "annual_income": float(annual_income),
            "home_ownership": home_ownership,
            "employment_years": float(employment_years),
            "loan_intent": loan_intent,
            "loan_amount": float(loan_amount),
            "interest_rate": float(interest_rate),
            "loan_percent_income": round(float(loan_amount) / float(annual_income), 3),
            "cb_person_default_on_file": default_on_file,
            "credit_score": int(credit_score),
            "credit_history_years": int(credit_history_years),
        }])

        pred_class = model.predict(applicant_payload)[0]
        pred_proba = model.predict_proba(applicant_payload)[0]
        good_credit_prob = pred_proba[1]
        bad_credit_prob = pred_proba[0]

        st.session_state["last_prediction"] = True

        st.markdown("---")
        st.subheader("🎯 Automated Credit Assessment Decision")

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            if pred_class == 1:
                st.markdown(
                    f"""
                    <div class='decision-approved'>
                        <h2 style='margin-top:0; color:#065f46;'>🎉 APPROVED — GOOD CREDIT RATING</h2>
                        <p style='font-size:1.05rem; margin-bottom:8px;'>
                            The model estimates a <strong>{good_credit_prob * 100:.1f}% probability</strong> of reliable debt repayment.
                            This application falls comfortably within prime credit underwriting standards.
                        </p>
                        <span class='badge badge-success'>Low Default Probability</span>
                        <span class='badge badge-success'>Recommended for Standard Terms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class='decision-flagged'>
                        <h2 style='margin-top:0; color:#991b1b;'>⚠️ FLAGGED — HIGH CREDIT RISK (BAD CREDIT)</h2>
                        <p style='font-size:1.05rem; margin-bottom:8px;'>
                            The model estimates a <strong>{bad_credit_prob * 100:.1f}% probability</strong> of credit distress or default.
                            This profile does not satisfy standard unsecured lending parameters.
                        </p>
                        <span class='badge badge-danger'>High Default Risk</span>
                        <span class='badge badge-danger'>Requires Manual Underwriting or Collateral</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Key Financial Ratio Metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Loan-to-Income", f"{applicant_payload['loan_percent_income'].iloc[0]*100:.1f}%")
            with mcol2:
                st.metric("Credit Score", f"{credit_score}")
            with mcol3:
                st.metric("Default Record", "Yes" if default_on_file == "Y" else "None")

            # Underwriting Risk Breakdown & Actionable Advice
            st.markdown("#### 💡 Underwriting Insights & Recommendations")
            risk_notes = []
            if credit_score < 620:
                risk_notes.append("• **Low FICO Score**: Scores below 620 significantly reduce approval probability.")
            if applicant_payload['loan_percent_income'].iloc[0] > 0.35:
                risk_notes.append("• **High Debt Burden**: Loan principal represents over 35% of annual gross earnings.")
            if default_on_file == "Y":
                risk_notes.append("• **Credit Bureau Default**: Historical default on record is one of the strongest negative predictors.")
            if employment_years < 2.0:
                risk_notes.append("• **Short Employment Duration**: Less than 2 years of steady employment history raises stability concerns.")

            if risk_notes:
                st.warning("Key Identified Risk Factors:\n\n" + "\n".join(risk_notes))
            else:
                st.success("✅ **Profile Strengths**: Stable income, strong credit bureau history, and sustainable loan-to-income ratio.")

        with res_col2:
            # Interactive Plotly Gauge for Credit Repayment Confidence
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=round(good_credit_prob * 100, 1),
                    title={"text": "Credit Confidence Score (P[Good Credit])", "font": {"size": 17}},
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#10b981" if pred_class == 1 else "#ef4444"},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [0, 40], "color": "#fee2e2"},
                            {"range": [40, 70], "color": "#fef3c7"},
                            {"range": [70, 100], "color": "#d1fae5"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": 50.0,
                        },
                    },
                )
            )
            gauge_fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)


# =========================================================
# TAB 2: Model Performance & Metrics
# =========================================================
with tab_perf:
    st.subheader("📈 Machine Learning Performance & Validation")
    st.caption("Detailed statistical evaluation of the Random Forest Classifier on the 20% held-out test split.")

    if metrics:
        # Top Metrics Grid
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%", help="Total correct predictions / all predictions.")
        kpi2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="Area Under the Receiver Operating Characteristic curve.")
        kpi3.metric("Precision", f"{metrics['precision']*100:.2f}%", help="True Good predictions / all predicted Good.")
        kpi4.metric("Recall", f"{metrics['recall']*100:.2f}%", help="Correctly identified Good borrowers / actual Good.")
        kpi5.metric("F1-Score", f"{metrics['f1_score']:.3f}", help="Harmonic mean of Precision and Recall.")

        st.markdown("---")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 🧩 Confusion Matrix")
            st.caption("Distribution of correct vs incorrect classifications on test data.")
            cm = metrics["confusion_matrix"]
            
            # Plotly Confusion Matrix Heatmap
            cm_z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
            x_labels = ["Predicted: Bad", "Predicted: Good"]
            y_labels = ["Actual: Bad", "Actual: Good"]

            cm_fig = px.imshow(
                cm_z,
                text_auto=True,
                x=x_labels,
                y=y_labels,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
            )
            cm_fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(cm_fig, use_container_width=True)

            st.info(
                f"""
                - **True Negatives** (Bad identified as Bad): **{cm[0][0]}**
                - **False Positives** (Bad incorrectly approved): **{cm[0][1]}** *(Risky for lenders)*
                - **False Negatives** (Good incorrectly denied): **{cm[1][0]}** *(Lost business)*
                - **True Positives** (Good identified as Good): **{cm[1][1]}**
                """
            )

        with col_m2:
            st.markdown("#### 🏆 Feature Importance Rankings")
            st.caption("Top financial attributes driving the Random Forest model's decisions.")
            if features:
                feat_df = pd.DataFrame(features[:10])
                feat_df = feat_df.sort_values(by="importance", ascending=True)

                feat_fig = px.bar(
                    feat_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    text=[f"{v*100:.1f}%" for v in feat_df["importance"]],
                    color="importance",
                    color_continuous_scale="Viridis",
                    labels={"importance": "Relative Importance", "feature": "Feature"},
                )
                feat_fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                st.plotly_chart(feat_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📑 Detailed Classification Report")
        rep = metrics.get("classification_report", {})
        if rep:
            rep_data = []
            for label in ["0", "1"]:
                if label in rep:
                    name = "Bad Credit (0)" if label == "0" else "Good Credit (1)"
                    rep_data.append({
                        "Class": name,
                        "Precision": f"{rep[label]['precision']:.3f}",
                        "Recall": f"{rep[label]['recall']:.3f}",
                        "F1-Score": f"{rep[label]['f1-score']:.3f}",
                        "Support (Samples)": int(rep[label]['support']),
                    })
            st.dataframe(pd.DataFrame(rep_data), use_container_width=True, hide_index=True)


# =========================================================
# TAB 3: Dataset Explorer
# =========================================================
with tab_data:
    st.subheader("🔍 Synthetic Loan Applicant Dataset Explorer")
    st.caption("Explore the underlying training records, inspect distributions, and filter borrower profiles.")

    if df is not None:
        # Quick filter controls
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            status_filter = st.selectbox(
                "Filter by Credit Status:",
                ["All", "Good Credit (1)", "Bad Credit (0)"],
                index=0,
            )
        with f_col2:
            intent_filter = st.multiselect(
                "Filter by Loan Purpose:",
                options=list(df["loan_intent"].unique()),
                default=list(df["loan_intent"].unique())[:3],
            )
        with f_col3:
            housing_filter = st.multiselect(
                "Filter by Home Ownership:",
                options=list(df["home_ownership"].unique()),
                default=list(df["home_ownership"].unique()),
            )

        # Apply filters
        filtered_df = df.copy()
        if status_filter == "Good Credit (1)":
            filtered_df = filtered_df[filtered_df["credit_status"] == 1]
        elif status_filter == "Bad Credit (0)":
            filtered_df = filtered_df[filtered_df["credit_status"] == 0]

        if intent_filter:
            filtered_df = filtered_df[filtered_df["loan_intent"].isin(intent_filter)]
        if housing_filter:
            filtered_df = filtered_df[filtered_df["home_ownership"].isin(housing_filter)]

        st.write(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records:")
        st.dataframe(filtered_df.head(100), use_container_width=True, height=280)

        # Download button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Dataset (credit_data.csv)",
            data=csv_data,
            file_name="credit_data.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### 📊 Exploratory Visualizations")

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            # Distribution of Credit Score by Status
            fig_hist = px.histogram(
                df,
                x="credit_score",
                color="credit_status",
                barmode="overlay",
                nbins=35,
                title="Credit Bureau Score Distribution by Credit Status",
                color_discrete_map={0: "#ef4444", 1: "#10b981"},
                labels={"credit_status": "Credit Status", "credit_score": "FICO Score"},
            )
            fig_hist.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with vcol2:
            # Loan Amount vs Annual Income Scatter
            sample_scatter = df.sample(min(500, len(df)), random_state=42)
            fig_scatter = px.scatter(
                sample_scatter,
                x="annual_income",
                y="loan_amount",
                color="credit_status",
                title="Loan Amount vs. Annual Income (Sample)",
                color_discrete_map={0: "#ef4444", 1: "#10b981"},
                labels={"annual_income": "Annual Income ($)", "loan_amount": "Loan Amount ($)"},
                hover_data=["credit_score", "loan_percent_income"],
            )
            fig_scatter.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================
# TAB 4: Educational ML Guide & How It Works
# =========================================================
with tab_guide:
    st.subheader("📚 Beginner's Guide: How Machine Learning Powers Credit Scoring")

    st.markdown(
        """
        ### 1. What is Credit Scoring?
        Credit scoring is the process financial institutions and fintech lenders use to quantify the risk of a borrower defaulting on a debt obligation. 
        Traditional methods used rigid rules or basic linear scorecards (like classic FICO scoring). Modern lending leverages **Supervised Classification Machine Learning** to discover nuanced, non-linear relationships across many borrower attributes simultaneously.

        ---

        ### 2. The Machine Learning Pipeline Architecture
        To prevent **data leakage** and ensure seamless inference from web forms to the prediction engine, the model uses a unified `scikit-learn` Pipeline:
        """
    )

    st.code(
        """
        Pipeline(
            steps=[
                ('preprocessor', ColumnTransformer([
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])),
                ('classifier', RandomForestClassifier(n_estimators=150, max_depth=12))
            ]
        )
        """,
        language="python",
    )

    st.markdown(
        """
        - **StandardScaler**: Centers numerical features (e.g. Annual Income, Age, FICO score) around a mean of 0 and unit variance so no single high-magnitude number distorts calculations.
        - **OneHotEncoder**: Converts categorical labels (`home_ownership`, `loan_intent`, `cb_person_default_on_file`) into binary numerical indicator columns.
        - **Random Forest Classifier**: An ensemble of 150 randomized decision trees that vote on whether an applicant is prime or high-risk.

        ---

        ### 3. Understanding Evaluation Metrics in Credit Underwriting

        | Metric | Banking Meaning | Why It Matters |
        | :--- | :--- | :--- |
        | **Precision** | Out of all applicants predicted "Good", how many were truly creditworthy? | High precision protects lenders from charge-offs and unpaid debts. |
        | **Recall** | Out of all creditworthy applicants, how many did the bank successfully approve? | High recall ensures the lender doesn't turn away good, profitable borrowers. |
        | **ROC-AUC** | The model's ability to discriminate between good and bad credit across all threshold cutoffs. | Scores above 0.90 indicate excellent predictive separation power. |
        | **False Positive** | Approving an applicant who defaults. | **Costliest error** for banks (direct financial loss). |
        | **False Negative** | Rejecting an applicant who would have repaid. | Opportunity cost (lost interest income and customer goodwill). |

        ---

        ### 4. Ethical AI & Regulatory Compliance (Fair Lending)
        Automated credit decision systems operate in highly regulated environments:
        - **Equal Credit Opportunity Act (ECOA)**: Forbids using sensitive demographic variables like race, gender, religion, national origin, or marital status in credit decisions.
        - **Adverse Action Disclosures**: Under the **Fair Credit Reporting Act (FCRA)**, if a model rejects an applicant, the lender must provide explicit key factors that caused the denial (which is why our app provides feature importance and individual risk factor explanations).
        """
    )

st.markdown("---")
st.caption("Credit Scoring Prediction System • Built with Python, scikit-learn, pandas & Streamlit • For educational & demonstration purposes.")
