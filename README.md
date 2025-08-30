# Stroke Prediction Project

## 📌 Project Overview
This project focuses on predicting the likelihood of a stroke based on healthcare-related features such as age, hypertension, heart disease, average glucose level, BMI, work type, smoking status, and more.  
The dataset used is **Healthcare Stroke Dataset**.

The workflow includes:
- Data Preprocessing  
- Exploratory Data Analysis (EDA)  
- Model Training & Evaluation  
- Model Selection (choosing the best performing algorithm)  
- Deployment of a **Streamlit Web App** for real-time predictions  

---

## 📂 Dataset
- **Source:** Healthcare Stroke Dataset  
- **Target Variable:** `stroke` (1 = stroke occurred, 0 = no stroke)  
- **Features Used:**
  - Age  
  - Hypertension  
  - Heart Disease  
  - Average Glucose Level  
  - BMI  
  - Gender  
  - Marital Status  
  - Work Type  
  - Residence Type  
  - Smoking Status  

---

## 🔎 Steps Performed

### 1. Data Preprocessing
- Handled missing values  
- Converted categorical variables into numerical (One-Hot Encoding)  
- Normalized numerical features  

### 2. Exploratory Data Analysis (EDA)
- Distribution of stroke cases across features  
- Correlation analysis between health factors and stroke occurrence  
- Visualizations to understand feature importance  

### 3. Model Training & Evaluation
- Implemented and compared multiple ML models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- Evaluated models using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - ROC-AUC  

### 4. Model Selection
- Compared all models  
- **Random Forest / XGBoost** showed the best performance  

### 5. Deployment
- Built a **Streamlit Web Application** (`app.py`)  
- The app allows users to input health details and get predictions on stroke risk  
- Displays probability of stroke occurrence  

---

## 🚀 Running the Project
APP LINK:
