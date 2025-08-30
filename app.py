import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("converted_data.csv")

feature_names = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

X = df[feature_names]
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

st.title("Stroke Prediction App")

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=1000.0, value=100.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Never_worked", "Private", "Self-employed", "children"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

input_data = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose_level, bmi,
                            1 if gender == "Male" else 0,
                            1 if ever_married == "Yes" else 0,
                            1 if work_type == "Never_worked" else 0,
                            1 if work_type == "Private" else 0,
                            1 if work_type == "Self-employed" else 0,
                            1 if work_type == "children" else 0,
                            1 if residence == "Urban" else 0,
                            1 if smoking_status == "formerly smoked" else 0,
                            1 if smoking_status == "never smoked" else 0,
                            1 if smoking_status == "smokes" else 0]],
                          columns=feature_names)

input_data_scaled = scaler.transform(input_data)

if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: {probability:.2f})")
