import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load trained model and preprocessing objects
model = joblib.load("logistic_regression_model.pkl")  # Either Random Forest or Logistic Regression
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
features = joblib.load("feature_names.pkl")  # List of columns used during training

# Title
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer data below to predict the likelihood of churn.")

# Sidebar Input
def user_input_form():
    st.sidebar.header("User Input")
    input_data = {}

    for col in features:
        if 'avg' in col or 'rate' in col or 'count' in col or 'total' in col:
            input_data[col] = st.sidebar.number_input(col, min_value=0.0, step=0.1)
        else:
            input_data[col] = st.sidebar.number_input(col, step=1)

    return pd.DataFrame([input_data])

# Get input and preprocess
input_df = user_input_form()

# Apply imputation and scaling
input_imputed = imputer.transform(input_df)
input_scaled = scaler.transform(input_imputed)

# Predict
pred_proba = model.predict_proba(input_scaled)[0][1]
pred_label = model.predict(input_scaled)[0]

# Display results
st.subheader("Prediction")
st.metric("Churn Probability", f"{pred_proba:.2%}")
st.write("Prediction:", "🔴 **Will Churn**" if pred_label == 1 else "🟢 **Will Stay**")

st.markdown("---")
st.markdown("✅ This prediction uses the model trained on the full dataset with SMOTE and scaling.")
