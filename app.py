"""Simple Streamlit interface for the educational diabetes model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model.joblib"

st.set_page_config(page_title="Diabetes Model Demo", page_icon="🩺")
st.title("Diabetes Model Demo")
st.caption("Educational demonstration only. This score is not a diagnosis or medical advice.")

if not MODEL_PATH.exists():
    st.error("The model has not been trained yet. In the VS Code terminal, run: python train.py")
    st.stop()

model = joblib.load(MODEL_PATH)
st.write("Enter example measurements, then select **Estimate**.")

with st.form("diabetes_inputs"):
    pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin thickness", min_value=0, max_value=100, value=25)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=28.5)
    pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.45)
    age = st.number_input("Age", min_value=21, max_value=120, value=35)
    submitted = st.form_submit_button("Estimate")

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }])
    score = model.predict_proba(row)[0, 1]
    st.metric("Model score for this example", f"{score:.1%}")
    st.info("This is an uncalibrated output from a limited historical dataset, not a personal medical risk estimate. Please discuss health concerns with a clinician.")
