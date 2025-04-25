import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing objects
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_order = joblib.load("feature_names.joblib")
col_info = joblib.load("col_info.joblib")  # contains feature UI data

# TENURE mapping (some values are duplicates)
tenure_order = {
    'A 0-3 month': 0,
    'B 3-6 month': 1,
    'C 6-9 month': 2,
    'D 9-12 month': 3,
    'E 12-15 month': 4,
    'F 15-18 month': 5,
    'G 18-21 month': 6,
    'H 21-24 month': 7,
    'I 18-21 month': 6,
    'J 21-24 month': 7,
    'K > 24 month': 8
}

# App title
st.title("📱 Expresso Churn Prediction App")

# Input form
with st.form("user_input_form"):
    st.subheader("📋 Customer Information")

    input_data = {}

    # Categorical features
    for cat_col in ['TOP_PACK', 'REGION', 'TENURE']:
        options = col_info.get(cat_col, [])
        input_data[cat_col] = st.selectbox(f"{cat_col}", options)

    # Numerical features
    for num_col in ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 
                    'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
                    'REGULARITY', 'FREQ_TOP_PACK']:
        min_val, max_val = col_info[num_col]
        default_val = (min_val + max_val) / 2
        input_data[num_col] = st.slider(num_col, float(min_val), float(max_val), float(default_val))

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    # Replace tenure with its mapped value
    input_data['TENURE'] = tenure_order.get(input_data['TENURE'], 0)

    # Assemble DataFrame
    input_df = pd.DataFrame([input_data])[feature_order]

    # Apply scaling to numerical features only
    num_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 
                'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
                'REGULARITY', 'FREQ_TOP_PACK']
    
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    st.subheader("📊 Prediction Result")
    st.write("Churn Prediction:", "Yes ✅" if prediction == 1 else "No ❌")
    st.write(f"Churn Probability: {probability:.2%}")
