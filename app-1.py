import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load saved model & references
model = joblib.load("log_reg_over1.joblib")
feature_names = joblib.load("feature_names.joblib")
col_info = joblib.load("unique_elements_dict1.joblib")  # Contains options like TENURE, REGION, TOP_PACK

# -----------------------
# UI: User Input Form
# -----------------------
with st.form("predict_form"):
    st.title("📱 Churn Prediction (Expresso Users)")

    REGION = st.selectbox("REGION", col_info["REGION"])
    
    TENURE = st.selectbox("TENURE", col_info["TENURE"])

    
    ARPU_SEGMENT = st.slider("ARPU_SEGMENT", col_info["TENURE"])
    
    REGULARITY = st.slider("REGULARITY", 0.0, 1.0, 0.5)
    

    submitted = st.form_submit_button("Predict")

# -----------------------
# Data Transformation
# -----------------------
if submitted:
    # 1. Raw input
    df = pd.DataFrame([{
        "REGION": REGION, "TENURE": TENURE,
         "ARPU_SEGMENT": ARPU_SEGMENT,"REGULARITY": REGULARITY, 
    }])

    # Frequency encode REGION
region_freq = df['REGION'].value_counts(normalize=False)
df['REGION_FE'] = df['REGION'].map(region_freq)

# Normalize
scaler = MinMaxScaler()
df['REGION_FE'] = scaler.fit_transform(df[['REGION_FE']])

# Ordinal encode TENURE (better)
tenure_order = ['A < 1 month', 'B 1-3 month', 'C 3-6 month', 'D 6-9 month',
                'E 9-12 month', 'F 12-15 month', 'G 15-18 month', 'H 18-21 month',
                'I 21-24 month', 'J 24 month', 'K > 24 month']
df['TENURE_OE'] = df['TENURE'].astype(pd.CategoricalDtype(categories=tenure_order, ordered=True)).cat.codes
from sklearn.preprocessing import StandardScaler
num_cols = ['ARPU_SEGMENT', 'REGULARITY']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df 

    # -----------------------
    # Predict & Display
    # -----------------------
prediction = model.predict(df)[0]
prob = model.predict_proba(df)[0][1]

st.success("✅ Churn" if prediction == 1 else "❌ Not Churn")
st.info(f"📈 Churn Probability: {prob:.2%}")
