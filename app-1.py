import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load saved model & references
model = joblib.load("clf.joblib")

col_info = joblib.load("unique_elements_dict2.joblib")  # Contains options like TENURE, REGION, TOP_PACK

# -----------------------
# UI: User Input Form
# -----------------------
with st.form("predict_form"):
    st.title("📱 Churn Prediction (Expresso Users)")

    REGION = st.selectbox("REGION", col_info["REGION"])
    TENURE = st.selectbox("TENURE", col_info["TENURE"])
    
    MONTANT = st.slider("MONTANT", float(col_info["MONTANT"]["min"]), float(col_info["MONTANT"]["max"]))
    FREQUENCE_RECH = st.slider("FREQUENCE_RECH", float(col_info["FREQUENCE_RECH"]["min"]), float(col_info["FREQUENCE_RECH"]["max"]))
    REVENUE = st.slider("REVENUE", float(col_info["REVENUE"]["min"]), float(col_info["REVENUE"]["max"]))
    ARPU_SEGMENT = st.slider("ARPU_SEGMENT", float(col_info["ARPU_SEGMENT"]["min"]), float(col_info["ARPU_SEGMENT"]["max"]))
    FREQUENCE = st.slider("FREQUENCE", float(col_info["FREQUENCE"]["min"]), float(col_info["FREQUENCE"]["max"]))
    DATA_VOLUME = st.slider("DATA_VOLUME", float(col_info["DATA_VOLUME"]["min"]), float(col_info["DATA_VOLUME"]["max"]))
    ON_NET = st.slider("ON_NET", float(col_info["ON_NET"]["min"]), float(col_info["ON_NET"]["max"]))
    ORANGE = st.slider("ORANGE", float(col_info["ORANGE"]["min"]), float(col_info["ORANGE"]["max"]))
    TIGO = st.slider("TIGO", float(col_info["TIGO"]["min"]), float(col_info["TIGO"]["max"]))
    REGULARITY = st.slider("REGULARITY", float(col_info["REGULARITY"]["min"]), float(col_info["REGULARITY"]["max"]))

    submitted = st.form_submit_button("Predict")

# -----------------------
# Data Transformation
# -----------------------
if submitted:
    # 1. Raw input to DataFrame
    df = pd.DataFrame([{
        "REGION": REGION,
        "TENURE": TENURE,
        "MONTANT": MONTANT,
        "FREQUENCE_RECH": FREQUENCE_RECH,
        "REVENUE": REVENUE,
        "ARPU_SEGMENT": ARPU_SEGMENT,
        "FREQUENCE": FREQUENCE,
        "DATA_VOLUME": DATA_VOLUME,
        "ON_NET": ON_NET,
        "ORANGE": ORANGE,
        "TIGO": TIGO,
        "REGULARITY": REGULARITY,
    }])

    # Frequency encode REGION
# Frequency encode REGION
region_freq = df['REGION'].value_counts(normalize=False)
df['REGION_FE'] = df['REGION'].map(region_freq)

# Normalize
scaler = MinMaxScaler()
df['REGION_FE'] = scaler.fit_transform([['REGION_FE']])

# Ordinal encode TENURE (better)
tenure_order = ['A < 1 month', 'B 1-3 month', 'C 3-6 month', 'D 6-9 month',
                'E 9-12 month', 'F 12-15 month', 'G 15-18 month', 'H 18-21 month',
                'I 21-24 month', 'J 24 month', 'K > 24 month']
df['TENURE_OE'] = df['TENURE'].astype(pd.CategoricalDtype(categories=tenure_order, ordered=True)).cat.codes

# Frequency encode TOP_PACK
top_pack_freq = df['TOP_PACK'].value_counts()
df['TOP_PACK_FE'] = df['TOP_PACK'].map(top_pack_freq)

# Normalize the frequency encoding to [0,1]
scaler = MinMaxScaler()
df['TOP_PACK_FE'] = scaler.fit_transform(df[['TOP_PACK_FE']])

# Drop original column
df.drop(columns=['TOP_PACK'], inplace=True)
df.drop(columns=['REGION', 'TENURE'], inplace=True)
from sklearn.preprocessing import StandardScaler
# Columns to scale (excluding target and already normalized/ordinal encoded ones)
num_cols_to_scale = [
    'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
    'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK'
]

# Initialize scaler and fit-transform
scaler = StandardScaler()
df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale]) 

    # -----------------------
    # Predict & Display
    # -----------------------
prediction = model.predict(df)[0]
prob = model.predict_proba(df)[0][1]

st.success("✅ Churn" if prediction == 1 else "❌ Not Churn")
st.info(f"📈 Churn Probability: {prob:.2%}")
