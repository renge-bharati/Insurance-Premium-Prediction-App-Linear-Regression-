import streamlit as st
import pandas as pd
import numpy as np
import os

# ----------------------------
# Load trained model
# ----------------------------
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")

st.title("💰 Insurance Premium Prediction App")
st.write("Predict insurance charges based on customer details using a trained Linear Regression model.")

try:
    model = pd.read_pickle(model_path)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# ----------------------------
# User input section
# ----------------------------
st.header("Enter Customer Details")

age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ----------------------------
# Preprocessing input
# ----------------------------
# Convert categorical data to numerical or one-hot encode (depends on how model was trained)
# Here is a simple placeholder version
smoker_val = 1 if smoker == "Yes" else 0
sex_val = 1 if sex == "Male" else 0

# You may need to one-hot encode the region variable if your model expects it.
# For simplicity, let’s just map it to numbers:
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_val = region_map[region]

# Create a single-row input array
input_data = np.array([[age, bmi, children, smoker_val, sex_val, region_val]])

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Premium"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💵 Estimated Insurance Premium: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and a Linear Regression model.")
