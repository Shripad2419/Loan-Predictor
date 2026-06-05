import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
 
# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💳",
    layout="wide",
)
 
st.title("💳 Loan Default Risk Predictor")
st.markdown(
    "Predict loan default risk using XGBoost + SHAP explainability. "
    "Enter details manually **or** upload a CSV of loan applications."
)
 
# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("models/model.pkl"):
        st.error("⚠️ Model not found! Please run `python src/train.py` first.")
        st.stop()
    model    = joblib.load("models/model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features
 
model, feature_names = load_model()
 
# ── SHAP explainer (cached) ──────────────────────────────────
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)
 
explainer = get_explainer(model)
 
# ── Mode selector ─────────────────────────────────────────────
st.sidebar.header("⚙️ Input Mode")
mode = st.sidebar.radio(
    "Choose how to provide data:",
    ["✏️ Manual entry", "📂 Upload CSV file"],
    index=0,
)
st.sidebar.divider()
 
 
# ════════════════════════════════════════════════════════════
#  MODE 1 — MANUAL ENTRY (sliders)
# ════════════════════════════════════════════════════════════
if mode == "✏️ Manual entry":
 
    st.sidebar.header("📋 Loan Application Details")
 
    duration      = st.sidebar.slider("Loan duration (months)", 6, 72, 24)
    credit_amount = st.sidebar.slider("Credit amount (€)", 500, 20000, 5000, step=500)
    age           = st.sidebar.slider("Applicant age", 18, 75, 35)
    installment   = st.sidebar.slider("Installment rate (% of income)", 1, 4, 2)
    residence     = st.sidebar.slider("Years at current residence", 1, 4, 2)
    num_credits   = st.sidebar.slider("Existing credits at this bank", 1, 4, 1)
    dependents    = st.sidebar.slider("Number of dependents", 1, 2, 1)
 
    # Build full feature vector (zeros for columns not in sliders)
    input_values = {f: 0 for f in feature_names}
    input_values["duration"]               = duration
    input_values["credit_amount"]          = credit_amount
    input_values["age"]                    = age
    input_values["installment_commitment"] = installment
    input_values["residence_since"]        = residence
    input_values["existing_credits"]       = num_credits
    input_values["num_dependents"]         = dependents
    input_df = pd.DataFrame([input_values])
 
    # ── Prediction ────────────────────────────────────────────
    prob     = model.predict_proba(input_df)[0][1]
    risk_pct = f"{prob:.1%}"
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", risk_pct)
    col2.metric("Loan Duration",       f"{duration} months")
    col3.metric("Credit Amount",       f"€{credit_amount:,}")
 
    if prob >= 0.7:
        st.error(f"🔴 HIGH RISK — {risk_pct} probability of default. Loan not recommended.")
    elif prob >= 0.4:
        st.warning(f"🟡 MEDIUM RISK — {risk_pct} probability of default. Review carefully.")
    else:
        st.success(f"🟢 LOW RISK — {risk_pct} probability of default. Loan likely safe.")
 
    # ── SHAP Explainability ───────────────────────────────────
    st.subheader("🔍 Why did the model make this decision?")
    st.caption("SHAP values show which features pushed the risk up or down.")
 
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, color_bar=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # ── Input summary table ───────────────────────────────────
    st.subheader("📊 Input Summary")
    display_df = pd.DataFrame({
        "Feature": ["Duration", "Credit Amount", "Age",
                    "Installment Rate", "Residence (yrs)",
                    "Existing Credits", "Dependents"],
        "Value":   [f"{duration} months", f"€{credit_amount:,}",
                    f"{age} yrs", f"{installment}%",
                    f"{residence} yrs", num_credits, dependents],
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
 
 
# ════════════════════════════════════════════════════════════
#  MODE 2 — UPLOAD CSV FILE
# ════════════════════════════════════════════════════════════
else:
    st.subheader("📂 Upload Loan Applications Dataset")
 
    # ── Instructions ─────────────────────────────────────────
    with st.expander("📋 How to prepare your CSV file", expanded=True):
        st.markdown("""
**Your CSV must contain these columns** (same names, any order):
 
| Column | Description | Example |
|--------|-------------|---------|
| `duration` | Loan duration in months | 24 |
| `credit_amount` | Credit amount in € | 5000 |
| `age` | Applicant age | 35 |
| `installment_commitment` | Installment rate % of income (1–4) | 2 |
| `residence_since` | Years at current residence (1–4) | 2 |
| `existing_credits` | Number of existing credits at bank | 1 |
| `num_dependents` | Number of dependents | 1 |
 
All other columns are optional — missing columns default to 0.
 
👇 **Download a sample CSV** to see the exact format:
        """)
 
        # Generate and offer a sample CSV for download
        sample_data = pd.DataFrame({
            "duration":               [12, 24, 36, 48, 6],
            "credit_amount":          [1500, 8000, 3500, 12000, 500],
            "age":                    [28, 45, 33, 52, 22],
            "installment_commitment": [2, 3, 1, 4, 2],
            "residence_since":        [2, 4, 1, 3, 1],
            "existing_credits":       [1, 2, 1, 3, 1],
            "num_dependents":         [1, 2, 1, 1, 1],
        })
        csv_bytes = sample_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download sample_loans.csv",
            data=csv_bytes,
            file_name="sample_loans.csv",
            mime="text/csv",
        )
 
    # ── File uploader ─────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload your CSV file here",
        type=["csv"],
        help="CSV must contain at minimum: duration, credit_amount, age",
    )
 
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: **{uploaded_file.name}** — {len(df_upload)} row(s) found")
 
            # Preview
            st.subheader("👀 Preview of uploaded data")
            st.dataframe(df_upload.head(10), use_container_width=True)
 
            # ── Validate required columns ─────────────────────
            required_cols = ["duration", "credit_amount", "age"]
            missing_required = [c for c in required_cols if c not in df_upload.columns]
 
            if missing_required:
                st.error(
                    f"❌ Missing required columns: **{', '.join(missing_required)}**. "
                    "Please fix your CSV and re-upload."
                )
                st.stop()
 
            # ── Build model input (fill missing feature cols with 0) ──
            input_df = pd.DataFrame(0, index=range(len(df_upload)), columns=feature_names)
            for col in feature_names:
                if col in df_upload.columns:
                    input_df[col] = df_upload[col].fillna(0).values
 
            # ── Run predictions ───────────────────────────────
            with st.spinner("Running predictions..."):
                probs      = model.predict_proba(input_df)[:, 1]
                preds      = model.predict(input_df)
 
            # ── Build results dataframe ───────────────────────
            results_df = df_upload.copy()
            results_df["Default Probability (%)"] = (probs * 100).round(1)
            results_df["Prediction"]              = np.where(preds == 1, "❌ Default", "✅ Repay")
            results_df["Risk Level"] = pd.cut(
                probs,
                bins=[-0.01, 0.4, 0.7, 1.01],
                labels=["🟢 Low", "🟡 Medium", "🔴 High"],
            )
 
            # ── Summary stats ─────────────────────────────────
            st.subheader("📊 Prediction Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Applications", len(results_df))
            c2.metric("Likely to Repay ✅",  int((preds == 0).sum()))
            c3.metric("Likely to Default ❌", int((preds == 1).sum()))
            c4.metric("Avg Default Risk",     f"{probs.mean():.1%}")
 
            # Risk level breakdown bar chart
            st.subheader("📈 Risk Level Distribution")
            risk_counts = results_df["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            st.bar_chart(risk_counts.set_index("Risk Level"))
 
            # ── Full results table ────────────────────────────
            st.subheader("📋 All Predictions")
            st.dataframe(
                results_df.sort_values("Default Probability (%)", ascending=False),
                use_container_width=True,
            )
 
            # ── Download results as CSV ───────────────────────
            output_csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=output_csv,
                file_name="loan_predictions.csv",
                mime="text/csv",
            )
 
            # ── SHAP for first row ────────────────────────────
            st.subheader("🔍 SHAP Explanation — First Application")
            st.caption("Feature importance for the first row in your uploaded file.")
            first_row     = input_df.iloc[[0]]
            shap_values   = explainer.shap_values(first_row)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_values, first_row, plot_type="bar", show=False, color_bar=False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
 
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Make sure your CSV is properly formatted. Download the sample file above for reference.")
 
    else:
        st.info("👆 Upload a CSV file above to get batch predictions across multiple loan applications.")
 
 
# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Model: XGBoost | Dataset: German Credit (OpenML) | Explainability: SHAP | Built with Streamlit")