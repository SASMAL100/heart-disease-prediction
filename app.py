import streamlit as st
import joblib
import numpy as np

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Risk Prediction")
st.write("Enter patient clinical details to predict heart disease risk.")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0)
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=normal, 2=fixed, 3=reversible)", [1, 2, 3])

sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

if st.button("Predict"):
    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg,
                             thalach, exang_val, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"High risk of heart disease (probability: {probability:.1%})")
    else:
        st.success(f"Low risk of heart disease (probability: {probability:.1%})")
