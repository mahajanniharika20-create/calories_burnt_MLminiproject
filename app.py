import json

import joblib
import pandas as pd
import streamlit as st

from train_model import (FEATURES, LINEAR_PATH, META_PATH, MODEL_PATH,
                         VALID_RANGES, validate_input)

# Page config
st.set_page_config(page_title="Calories Burnt Predictor", layout="centered")

st.title("Calories Burnt Prediction System")
st.write("Enter your exercise details below to estimate total calories burnt.")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_trained_ranges():
    """Ranges the main model was trained on (saved by train_model.py)."""
    try:
        return json.loads(META_PATH.read_text()).get("trained_ranges", {})
    except FileNotFoundError:
        return {}


try:
    model = load_model(str(MODEL_PATH))
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Run `python train_model.py` first.")
    st.stop()

# Rough fallback for inputs outside the training ranges (optional)
try:
    linear_model = load_model(str(LINEAR_PATH))
except FileNotFoundError:
    linear_model = None

trained = load_trained_ranges()


def hard(col):
    lo, hi = VALID_RANGES[col]
    return float(lo), float(hi)


def typical(col, default):
    """Default value that always sits inside the trained range."""
    lo, hi = trained.get(col, hard(col))
    return float(min(max(default, lo), hi))


d_lo, d_hi = trained.get("Duration", hard("Duration"))
hr_lo, hr_hi = trained.get("Heart_Rate", hard("Heart_Rate"))
st.caption(
    f"The main model is most accurate within the ranges it was trained on "
    f"(e.g. duration {d_lo:.0f}–{d_hi:.0f} min, heart rate {hr_lo:.0f}–{hr_hi:.0f} bpm). "
    f"Outside them you'll get a clearly labelled rough estimate instead."
)

# Input form (bounds are the wide, physically plausible limits)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=int(hard("Age")[0]),
                              max_value=int(hard("Age")[1]), value=int(typical("Age", 25)))
        height = st.number_input("Height (cm)", min_value=hard("Height")[0],
                                 max_value=hard("Height")[1], value=typical("Height", 170.0))
        weight = st.number_input("Weight (kg)", min_value=hard("Weight")[0],
                                 max_value=hard("Weight")[1], value=typical("Weight", 70.0))

    with col2:
        duration = st.number_input("Duration (minutes)", min_value=hard("Duration")[0],
                                   max_value=hard("Duration")[1], value=typical("Duration", 20.0))
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=hard("Heart_Rate")[0],
                                     max_value=hard("Heart_Rate")[1],
                                     value=typical("Heart_Rate", 100.0))
        body_temp = st.number_input("Body Temp (°C)", min_value=hard("Body_Temp")[0],
                                     max_value=hard("Body_Temp")[1],
                                    value=typical("Body_Temp", 38.5))

    submit_button = st.form_submit_button("Predict Calories")

# Prediction logic
if submit_button:
    user_input = {
        "Gender": gender.lower(),
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }

    try:
        # ValueError -> impossible input; warnings -> outside trained ranges
        warnings = validate_input(user_input)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        st.stop()

    user_input["Gender"] = 0 if gender == "Male" else 1
    input_data = pd.DataFrame([user_input])[FEATURES]

    if not warnings:
        # Inside the training ranges: use the accurate XGBoost model
        prediction = max(0.0, float(model.predict(input_data)[0]))
        st.success(f"### Estimated Calories Burnt: **{prediction:.2f} kcal**")

    elif linear_model is not None:
        # Outside the training ranges: trees can't extrapolate, so use the
        # linear model and label the result as a rough estimate.
        prediction = max(0.0, float(linear_model.predict(input_data)[0]))
        st.warning(f"### Rough estimate: about **{prediction:.0f} kcal**")
        st.caption(
            "Some inputs are outside the data the main model was trained on, so this "
            "comes from a simpler linear model (less accurate, about 8 kcal off on average within "
            "the training ranges, and likely worse outside them). Treat it as a ballpark."
        )
        for w in warnings:
            st.write(f"- {w}")

    else:
        for w in warnings:
            st.error(w)
        st.error("No fallback model found. Run `python train_model.py` to create one.")
