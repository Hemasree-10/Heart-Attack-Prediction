import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('knn_scaler.pkl')

# Title of the app
st.title("Heart Attack Prediction App")

# Add a description
st.write("""
This app predicts the likelihood of a heart attack based on patient data.
Please enter the required information below.
""")

# Input fields for user data
st.sidebar.header("Patient Information")

# Numerical features
age = st.sidebar.slider("Age", 20, 100, 50)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 202, 150)
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.2, 1.0)

# Categorical features
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

# Map categorical inputs to numerical values
sex_map = {"Male": 1, "Female": 0}
chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
fasting_bs_map = {"No": 0, "Yes": 1}
resting_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exercise_angina_map = {"No": 0, "Yes": 1}
st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

# Convert user inputs to numerical values
sex = sex_map[sex]
chest_pain_type = chest_pain_map[chest_pain_type]
fasting_bs = fasting_bs_map[fasting_bs]
resting_ecg = resting_ecg_map[resting_ecg]
exercise_angina = exercise_angina_map[exercise_angina]
st_slope = st_slope_map[st_slope]

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    "Age": [age],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "MaxHR": [max_hr],
    "Oldpeak": [oldpeak],
    # One-hot encoded categorical features
    "Sex_M": [sex],
    "ChestPainType_ATA": [1 if chest_pain_type == 1 else 0],
    "ChestPainType_NAP": [1 if chest_pain_type == 2 else 0],
    "ChestPainType_TA": [1 if chest_pain_type == 0 else 0],
    "RestingECG_Normal": [1 if resting_ecg == 0 else 0],
    "RestingECG_ST": [1 if resting_ecg == 1 else 0],
    "ExerciseAngina_Y": [exercise_angina],
    "ST_Slope_Flat": [1 if st_slope == 1 else 0],
    "ST_Slope_Up": [1 if st_slope == 0 else 0]
})

# Display user inputs
st.subheader("Patient Data")
st.write(input_data)

# Scale the input data
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("**High Risk of Heart Attack**")
    else:
        st.write("**Low Risk of Heart Attack**")

    # Display prediction probability
    st.subheader("Prediction Probability")
    st.write(f"Probability of Heart Attack: {prediction_proba[0][1]:.2f}")
