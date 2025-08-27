
import streamlit as st
import joblib
import numpy as np


model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Prediction")


age = st.number_input("Age", min_value=1, max_value=120, value=63)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.selectbox("Chest pain type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP (mm Hg)", value=120)
chol = st.number_input("Cholesterol (mg/dl)", value=200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1 = true)", [0,1])
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
thalach = st.number_input("Max heart rate achieved", value=150)
exang = st.selectbox("Exercise induced angina (1 = yes)", [0,1])
oldpeak = st.number_input("ST depression (oldpeak)", value=1.0, format="%.2f")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of major vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (1=normal,2=fixed,3=reversible)", [1,2,3])

features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

if st.button("Predict"):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    st.success("Prediction: " + ("Heart Disease" if pred == 1 else "No Heart Disease"))
