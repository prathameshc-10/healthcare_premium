import streamlit as st
import joblib
import pandas as pd

# Load the brain
model = joblib.load('healthcare_model.pkl')

# Setup the page title
st.set_page_config(page_title="Healthcare Premium Predictor")
st.title("Healthcare Premium Predictor")
st.write("Enter your details below to estimate your annual insurance premium")

# Create the UI inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Select your age", 18, 100, 30)
    bmi = st.number_input(
        "Enter your BMI:",
        min_value=10.0,
        max_value=50.0,
        step=0.1,
        value=25.0
    )

with col2:
    smoker_choice = st.selectbox("Are you a smoker?", ["Yes", "No"])

# Prediction Button
if st.button("Calculate Premium"):
    # Convert inputs to match model format
    smoker = 1 if smoker_choice == "Yes" else 0
    input_df = pd.DataFrame([[age, bmi, smoker]], columns=["age", "bmi", "smoker"])

    # Get Prediction
    prediction = model.predict(input_df)[0]

    # Display Result
    st.success(f"Estimated Annual Premium: {prediction:,.2f} Rs")
    st.info("Note: This is basic ML model estimate based on provided training data.")
