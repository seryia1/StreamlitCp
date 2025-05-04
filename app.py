import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load model and reference values
model = joblib.load("log_reg_over1.joblib")
col_info = joblib.load("unique_elements_dict1.joblib")  # Includes REGION, TENURE, ARPU_SEGMENT, REGULARITY

# -----------------------
# UI: User Input Form
# -----------------------
st.title("📱 Churn Prediction (Expresso Users)")

with st.form("predict_form"):
    st.title("📱 Churn Prediction (Expresso Users)")

    REGION = st.selectbox("REGION", col_info["REGION"])
    TENURE = st.selectbox("TENURE", col_info["TENURE"])
    
    arpu_min = min(col_info["ARPU_SEGMENT"])
    arpu_max = max(col_info["ARPU_SEGMENT"])
    ARPU_SEGMENT = st.slider("ARPU_SEGMENT", min_value=float(arpu_min), max_value=float(arpu_max), value=float(arpu_min))

    reg_min = min(col_info["REGULARITY"])
    reg_max = max(col_info["REGULARITY"])
    REGULARITY = st.slider("REGULARITY", min_value=float(reg_min), max_value=float(reg_max), value=float(reg_min))

    submitted = st.form_submit_button("Predict")


# -----------------------
# Logic only runs if submitted
# -----------------------
if submitted:
    # 1. Create input DataFrame
    df = pd.DataFrame([{
        "REGION": REGION,
        "TENURE": TENURE,
        "ARPU_SEGMENT": float(ARPU_SEGMENT),
        "REGULARITY": float(REGULARITY)
    }])

    # 2. Frequency encode REGION (based on training reference, ideally)
    region_freq = df['REGION'].value_counts(normalize=False)
    df['REGION_FE'] = df['REGION'].map(region_freq)

    # Normalize REGION_FE
    scaler_region = MinMaxScaler()
    df['REGION_FE'] = scaler_region.fit_transform(df[['REGION_FE']])

    # 3. Ordinal encode TENURE
    tenure_order = ['A < 1 month', 'B 1-3 month', 'C 3-6 month', 'D 6-9 month',
                    'E 9-12 month', 'F 12-15 month', 'G 15-18 month', 'H 18-21 month',
                    'I 21-24 month', 'J 24 month', 'K > 24 month']
    df['TENURE_OE'] = df['TENURE'].astype(pd.CategoricalDtype(categories=tenure_order, ordered=True)).cat.codes

    # 4. Standardize numerical columns
    num_cols = ['ARPU_SEGMENT', 'REGULARITY']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Drop unused columns
    df.drop(columns=['REGION', 'TENURE'], inplace=True)

    # -----------------------
    # Predict & Show Results
    # -----------------------
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.success("✅ Churn" if prediction == 1 else "❌ Not Churn")
    st.info(f"📈 Churn Probability: {prob:.2%}")
