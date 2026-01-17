# 🏦 XGBoost Credit Risk Prediction App

An end-to-end Machine Learning project that predicts **credit default risk** using an optimized **XGBoost classifier**, deployed via **Streamlit**.

This project covers the full ML lifecycle:
- Feature engineering
- Handling class imbalance (SMOTE)
- Model tuning
- Model persistence
- Interactive web deployment

---

## 🚀 Features

- XGBoost model with high ROC-AUC (~0.946)
- Custom feature engineering (`age_group`)
- One-hot encoding using `pd.get_dummies`
- Scaler + model persistence
- Real-time predictions via Streamlit UI

---

## 📊 Model Performance (Best Model)

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.933 |
| Precision  | 0.932 |
| Recall     | 0.748 |
| F1-Score   | 0.830 |
| ROC-AUC    | 0.946 |

> The tuned SMOTE XGBoost model was selected due to **better recall and balanced F1-score**, making it suitable for imbalanced classification.

---

## 🧩 Feature Engineering

### Age Group Creation
```python
X["age_group"] = pd.cut(X["person_age"], bins=[18, 25, 35, 50, 100], labels=["18-25", "26-35", "36-50", "50+"])
```

---

### Encoding

Categorical features are encoded using:
```python
pd.get_dummies(drop_first=True)
```
The same encoded feature columns must be preserved during inference.

---

## 💾 Model Saving

```python
import joblib

joblib.dump(best_xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "features.pkl")
```

Saved objects:
- xgb_credit_risk_model.pkl → trained XGBoost model
- scaler.pkl → fitted scaler
- feature_columns.pkl → list of encoded feature names

---

## 🌐 Streamlit Deployment

```python
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Workflow:
- User inputs data via UI
- age_group is created dynamically
- Data is encoded using get_dummies
- Missing dummy columns are added (set to 0)
- Features are scaled
- Model predicts default risk

---

## 🛠️ Tech Stack

- Language: Python
- ML Model: XGBoost
- Data Processing: Pandas, NumPy
- Imbalance Handling: SMOTE
- Deployment: Streamlit
- Model Persistence: Joblib
- Data analysis: Matplotlib, Seaborn

---

## 🎯 Key Takeaways

- Encoding consistency is non-negotiable
- Feature engineering must be mirrored at inference
- Recall matters more than accuracy in risk modeling
- SMOTE + tuning gave the best real-world balance
