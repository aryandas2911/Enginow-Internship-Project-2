import streamlit as st
import pandas as pd
import numpy as np
import joblib

def get_age_group(age):
    if age < 25:
        return "18-25"
    elif age < 35:
        return "26-35"
    elif age < 50:
        return "36-50"
    else:
        return "50+"

model = joblib.load("models/xgb_credit_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("🏦 Credit Risk Prediction System")
st.write("Predict whether a loan applicant is **High Risk** or **Low Risk** "
        "using a trained Machine Learning model."
)

with st.form("credit_form"):
    st.subheader("Applicant Details")

    person_age = st.number_input("Age", min_value=18, max_value=100, value=18)
    person_income = st.number_input("Annual Income", min_value=0, value=10000)
    person_home_ownership = st.selectbox(
        "Home Ownership", ["RENT", "OWN", "MORTGAGE"]
    )
    person_emp_length = st.number_input(
        "Employment Length (years)", min_value=0.0, value=5.0
    )
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "PERSONAL",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
    )
    loan_grade = st.selectbox(
        "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
    )
    loan_amnt = st.number_input("Loan Amount", min_value=500, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input(
        "Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2
    )
    cb_person_default_on_file = st.selectbox(
        "Historical Default", ["Y", "N"]
    )

    cb_preson_cred_hist_length = st.number_input(
        "Credit History Length (years)", min_value=0, value=10
    )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_preson_cred_hist_length": cb_preson_cred_hist_length,
    }

    input_df = pd.DataFrame([input_data])

    input_df["age_group"] = get_age_group(person_age)

    input_df_encoded = pd.get_dummies(input_df)

    input_df_encoded = input_df_encoded.reindex(
        columns=feature_columns, fill_value=0
    )

    input_scaled = scaler.transform(input_df_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **High Risk Applicant**\n\n"
            f"Probability of High Risk: **{probability:.2f}**"
        )
    else:
        st.success(
            f"✅ **Low Risk Applicant**\n\n"
            f"Probability of High Risk: **{probability:.2f}**"
        )
