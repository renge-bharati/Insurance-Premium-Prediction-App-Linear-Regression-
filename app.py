import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Premium Prediction", page_icon="💰")

st.title("💰 Insurance Premium Prediction App (Linear Regression)")

# Load trained model
model = joblib.load("model/linear_model.pkl")

st.sidebar.header("Input Features")

# Input fields
age = st.sidebar.number_input("Age", 18, 100, 25)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])

# Convert categorical to numeric
smoker_val = 1 if smoker == "Yes" else 0

# Predict button
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([[age, bmi, children, smoker_val]],
                              columns=["age", "bmi", "children", "smoker"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Insurance Premium: ${prediction:.2f}")
