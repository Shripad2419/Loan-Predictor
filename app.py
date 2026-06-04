
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
    "Enter loan details in the sidebar to get an instant default risk prediction "
    "powered by XGBoost + SHAP explainability."
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("models/model.pkl"):
        st.error("Model not found! Please run `python src/train.py` first.")
        st.stop()
    model    = joblib.load("models/model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

model, features = load_model()

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header("📋 Loan Application Details")

duration      = st.sidebar.slider("Loan duration (months)", 6, 72, 24)
credit_amount = st.sidebar.slider("Credit amount (€)", 500, 20000, 5000, step=500)
age           = st.sidebar.slider("Applicant age", 18, 75, 35)
installment   = st.sidebar.slider("Installment rate (% of income)", 1, 4, 2)
residence     = st.sidebar.slider("Years at current residence", 1, 4, 2)
num_credits   = st.sidebar.slider("Existing credits at this bank", 1, 4, 1)
dependents    = st.sidebar.slider("Number of dependents", 1, 2, 1)

# Build input dataframe with default=0 for remaining features
input_values = {f: 0 for f in features}
input_values["duration"]               = duration
input_values["credit_amount"]          = credit_amount
input_values["age"]                    = age
input_values["installment_commitment"] = installment
input_values["residence_since"]        = residence
input_values["existing_credits"]       = num_credits
input_values["num_dependents"]         = dependents
input_df = pd.DataFrame([input_values])

# ── Prediction ───────────────────────────────────────────────
prob      = model.predict_proba(input_df)[0][1]
risk_pct  = f"{prob:.1%}"

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

# ── SHAP Explainability ──────────────────────────────────────
st.subheader("🔍 Why did the model make this decision?")
st.caption("SHAP values show which features pushed the risk up (red) or down (blue).")

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer   = get_explainer(model)
shap_values = explainer.shap_values(input_df)

fig, ax = plt.subplots(figsize=(10, 4))
shap.summary_plot(
    shap_values, input_df,
    plot_type="bar", show=False, color_bar=False
)
plt.tight_layout()
st.pyplot(fig)

# ── Feature table ────────────────────────────────────────────
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

st.divider()
st.caption("Model: XGBoost trained on German Credit Dataset | Explainability: SHAP | Built with Streamlit")
