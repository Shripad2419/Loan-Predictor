# 💳 Loan Default Risk Predictor

An end-to-end machine learning web app that predicts whether a loan applicant
is likely to default, with SHAP-based explainability.

**Live Demo:** [Loan Default Risk Prediction.streamlit.app]([https://share.streamlit.io](https://loan-predictor-yvvrdq7pvagjacmziarnlg.streamlit.app/))

---

## 🚀 How to Run Locally

### Step 1 — Clone and enter the project
```bash
git clone https://github.com/YOUR_USERNAME/loan-default-predictor.git
cd loan-default-predictor
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model
```bash
python src/train.py
```
This downloads the dataset and saves the trained model to `models/model.pkl`.

### Step 5 — Launch the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| XGBoost | Gradient boosting classifier |
| Scikit-learn | Data preprocessing + metrics |
| SHAP | Model explainability |
| Streamlit | Web dashboard |
| Joblib | Model serialisation |

---

## 📁 Project Structure
```
loan-default-predictor/
├── src/
│   └── train.py          # Train and save the model
├── models/               # Saved model files (auto-created)
├── app.py                # Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → New app → Select this repo
4. Set main file to `app.py` → Deploy

> **Note:** Add a startup command in Streamlit Cloud settings to train the model:
> `python src/train.py && streamlit run app.py`
> Or include pre-trained model files in the repo.

---

## 📊 Model Performance

- Dataset: German Credit Dataset (UCI / OpenML)
- Algorithm: XGBoost Classifier
- ROC-AUC: ~0.78
- Features: 20 financial and personal attributes
