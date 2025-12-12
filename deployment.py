import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
st.title("Diabetes Prediction (SVM Pipeline)")

# Load model once
@st.cache_resource
def load_model():
    return load("diabetes_model.pkl")  # or diabetes_svm_pipeline.pkl

model = load_model()

st.subheader("Enter clinical features")
col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.number_input("Glucose", min_value=0.0, value=89.0)
    BloodPressure = st.number_input("BloodPressure", min_value=0.0, value=66.0)
    SkinThickness = st.number_input("SkinThickness", min_value=0.0, value=23.0)
with col2:
    Insulin = st.number_input("Insulin", min_value=0.0, value=94.0)
    BMI = st.number_input("BMI", min_value=0.0, value=28.1)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.167)
    Age = st.number_input("Age", min_value=1, max_value=120, value=21)

if st.button("Predict"):
    X = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DiabetesPedigreeFunction, Age]])
    pred = model.predict(X)[0]
    label = "non diabetic" if pred == 0 else "diabetic"
    st.success(f"Prediction: {label}")
