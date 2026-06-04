

import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

print("Loading dataset...")

# Load German Credit dataset
data = fetch_openml("credit-g", version=1, as_frame=True)
df = data.frame

# Create target BEFORE encoding
y = (df["class"] == "bad").astype(int)

# Features
X = df.drop("class", axis=1)

# Encode categorical feature columns
for col in X.select_dtypes(include=["category", "object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("\nOverall target distribution:")
print(y.value_counts())

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))

try:
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.3f}")
except Exception as e:
    print(f"ROC-AUC could not be calculated: {e}")

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(list(X.columns), "models/features.pkl")

print("\nModel saved successfully!")
print("models/model.pkl")
print("models/features.pkl")

print("\nTraining complete!")
print("Run the app using:")
print("streamlit run app.py")