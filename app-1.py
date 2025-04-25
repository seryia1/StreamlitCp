import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and preprocessing objects
model = joblib.load("models/logistic_regression_model.joblib")
imputer = joblib.load("preprocessing/imputer.joblib")
scaler = joblib.load("preprocessing/scaler.joblib")
top_pack_freq = joblib.load("preprocessing/top_pack_freq.joblib")
feature_order = joblib.load("preprocessing/feature_order.joblib")  # list of 28 column names

# TENURE mapping
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

# Regions used during training
regions = [
    'Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Louga', 'Matam',
    'Saint-Louis', 'Sedhiou', 'Tambacounda', 'Thies', 'Ziguinchor', np.nan
]

# UI
st.title("📱 Expresso Churn Prediction App")

# 1. Collect User Input
st.header("🔍 Enter Customer Info")

tenure = st.selectbox("TENURE", list(tenure_order.keys()))
montant = st.number_input("MONTANT (recharge amount)", min_value=0.0)
frequence_rech = st.number_input("FREQUENCE_RECH (recharges/month)", min_value=0.0)
revenue = st.number_input("REVENUE", min_value=0.0)
arpu_segment = st.number_input("ARPU_SEGMENT", min_value=0.0)
frequence = st.number_input("FREQUENCE (activity days)", min_value=0.0)
data_volume = st.number_input("DATA_VOLUME (MB)", min_value=0.0)
on_net = st.number_input("ON_NET (on-net calls)", min_value=0.0)
orange = st.number_input("ORANGE (orange calls)", min_value=0.0)
tigo = st.number_input("TIGO (tigo calls)", min_value=0.0)
regularity = st.number_input("REGULARITY (active months)", min_value=0.0)
freq_top_pack = st.number_input("FREQ_TOP_PACK", min_value=0.0)
region = st.selectbox("REGION", regions)
top_pack = st.text_input("TOP_PACK")

# 2. Predict Button
if st.button("Predict Churn"):
    # 3. Preprocess Manually

    # Map tenure
    tenure_mapped = tenure_order.get(tenure, 0)

    # One-hot encode REGION
    region_encoded = {f"REGION_{r}": 0 for r in regions}
    region_encoded[f"REGION_{region}"] = 1 if f"REGION_{region}" in region_encoded else 0

    # Frequency encode TOP_PACK
    top_pack_value = top_pack_freq.get(top_pack, 0)
    top_pack_norm = scaler.transform([[top_pack_value]])[0][0]  # Normalize

    # Assemble final input
    input_data = {
        'TENURE': tenure_mapped,
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'REGULARITY': regularity,
        'FREQ_TOP_PACK': freq_top_pack,
        **region_encoded,
        'TOP_PACK_FE': top_pack_norm
    }

    # Ensure feature order
    input_df = pd.DataFrame([input_data])[feature_order]

    # Impute any missing values (if applicable)
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Predict
    prediction = model.predict(input_df_imputed)[0]
    probability = model.predict_proba(input_df_imputed)[0][1]

    # Output
    st.subheader("📊 Prediction Result")
    st.write("Churn Prediction:", "Yes ✅" if prediction == 1 else "No ❌")
    st.write(f"Churn Probability: {probability:.2%}")
