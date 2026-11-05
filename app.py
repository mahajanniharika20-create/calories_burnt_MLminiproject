import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(page_title="Calories Burnt Predictor", page_icon="🔥", layout="centered")

# Title and Description
st.title("🔥 Calories Burnt Prediction System")
st.write("Enter your exercise details below to estimate total calories burnt.")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("calories_model.pkl")

model = load_model()

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

    with col2:
        duration = st.number_input("Duration (minutes)", min_value=1.0, max_value=180.0, value=30.0)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=60.0, max_value=220.0, value=100.0)
        body_temp = st.number_input("Body Temp (°C)", min_value=35.0, max_value=42.0, value=38.5)

    submit_button = st.form_submit_button("Predict Calories")

# Prediction Logic
if submit_button:
    # Encode Gender (Male -> 0, Female -> 1)
    gender_encoded = 0 if gender == "Male" else 1

    input_data = pd.DataFrame([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]],
                              columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Calories Burnt: **{prediction:.2f} kcal**")