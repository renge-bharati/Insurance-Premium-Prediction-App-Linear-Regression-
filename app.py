import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Premium Prediction", page_icon="💰", layout="centered")

st.title("💰 Insurance Premium Prediction App (Linear Regression)")
st.subheader("Predict health insurance cost using a pre-trained base model")

# Load your base model
model = joblib.load("base_model.pkl")

# Sidebar inputs
st.sidebar.header("Enter Details")

age = st.sidebar.number_input("Age", 18, 100, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])

# Convert categorical to numeric
smoker_val = 1 if smoker == "Yes" else 0

# Prepare input
input_df = pd.DataFrame([[age, bmi, children, smoker_val]], columns=["age", "bmi", "children", "smoker"])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Insurance Premium: ${prediction:.2f}")

