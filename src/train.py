"""
Train the loan default prediction model.
Run this first before launching the Streamlit app.
Usage: python src/train.py
"""

import pandas as pd
import joblib
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

print("Loading dataset...")
# Fetch German credit dataset (no Kaggle account needed)
data = fetch_openml("credit-g", version=1, as_frame=True)
df = data.frame

# Encode categorical columns
for col in df.select_dtypes("category").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Target: 'class' column (good=0, bad=1)
X = df.drop("class", axis=1)
y = (df["class"] == "bad").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.3f}")

# Save model and feature list
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")
print("\nModel saved to models/model.pkl")
print("Training complete! Now run: streamlit run app.py")
