import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🚀 Load Trained Model and Features
@st.cache_resource
def load_model():
    model = joblib.load("chronic_disease_model (2).pkl")
    features = joblib.load("model_features (2).pkl")
    return model, features

model, feature_cols = load_model()

# 🖼️ Page Setup
st.set_page_config(page_title="Chronic Disease Predictor", layout="centered")
st.title("🔬 Chronic Disease Risk Predictor")
st.markdown("Upload a trained model or use user input to predict risk of **Diabetes, Heart Disease, or Cancer**.")

# 👤 User Input Section
st.header("🧑‍⚕️ Enter Patient Information")

def user_input():
    inputs = {}
    for col in feature_cols:
        if col.lower() == 'age':
            inputs[col] = st.slider('Age', 20, 90, 40)
        elif col.lower() == 'gender':
            gender = st.selectbox("Gender", ['Male', 'Female'])
            inputs[col] = 1 if gender == 'Male' else 0
        elif col.lower() == 'bmi':
            inputs[col] = st.slider('BMI', 10.0, 50.0, 25.0)
        elif col.lower() == 'glucose':
            inputs[col] = st.slider('Glucose Level', 60, 250, 100)
        elif col.lower() == 'cholesterol':
            inputs[col] = st.slider('Cholesterol', 100, 400, 200)
        elif 'pressure' in col.lower():
            inputs[col] = st.slider('Blood Pressure', 50, 200, 120)
        else:
            inputs[col] = st.number_input(col, value=0.0)
    return pd.DataFrame([inputs])

# ⏳ Get Input
input_df = user_input()

# 🧠 Predict Risk
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction Result")
    for i, disease in enumerate(['Diabetes', 'Heart Disease', 'Cancer']):
        result = prediction
        

    st.write(f"🩺 **{disease}**: {'❗ At Risk' if result else '✅ Not at Risk'}")

        

# 📄 Optional: Show Input Data
if st.checkbox("Show Entered Data"):
    st.dataframe(input_df)
