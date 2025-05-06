import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load saved model & references
model = joblib.load("clf.joblib")

col_info = joblib.load("unique_elements_dict2.joblib")  # Contains options like TENURE, REGION, TOP_PACK

# UI Form
with st.form("predict_form"):
    st.title("📱 Churn Prediction (Expresso Users)")

    REGION = st.selectbox("REGION", col_info["REGION"])
    TENURE = st.selectbox("TENURE", col_info["TENURE"])

    MONTANT = st.slider("MONTANT", float(min(col_info["MONTANT"])), float(max(col_info["MONTANT"])), float(min(col_info["MONTANT"])))
    FREQUENCE_RECH = st.slider("FREQUENCE_RECH", float(min(col_info["FREQUENCE_RECH"])), float(max(col_info["FREQUENCE_RECH"])), float(min(col_info["FREQUENCE_RECH"])))
    REVENUE = st.slider("REVENUE", float(min(col_info["REVENUE"])), float(max(col_info["REVENUE"])), float(min(col_info["REVENUE"])))
    ARPU_SEGMENT = st.slider("ARPU_SEGMENT", float(min(col_info["ARPU_SEGMENT"])), float(max(col_info["ARPU_SEGMENT"])), float(min(col_info["ARPU_SEGMENT"])))
    FREQUENCE = st.slider("FREQUENCE", float(min(col_info["FREQUENCE"])), float(max(col_info["FREQUENCE"])), float(min(col_info["FREQUENCE"])))
    DATA_VOLUME = st.slider("DATA_VOLUME", float(min(col_info["DATA_VOLUME"])), float(max(col_info["DATA_VOLUME"])), float(min(col_info["DATA_VOLUME"])))
    ON_NET = st.slider("ON_NET", float(min(col_info["ON_NET"])), float(max(col_info["ON_NET"])), float(min(col_info["ON_NET"])))
    ORANGE = st.slider("ORANGE", float(min(col_info["ORANGE"])), float(max(col_info["ORANGE"])), float(min(col_info["ORANGE"])))
    TIGO = st.slider("TIGO", float(min(col_info["TIGO"])), float(max(col_info["TIGO"])), float(min(col_info["TIGO"])))
    REGULARITY = st.slider("REGULARITY", float(min(col_info["REGULARITY"])), float(max(col_info["REGULARITY"])), float(min(col_info["REGULARITY"])))
    TOP_PACK = st.selectbox("TOP_PACK", col_info["TOP_PACK"])
    FREQ_TOP_PACK = st.slider("FREQ_TOP_PACK", float(min(col_info["FREQ_TOP_PACK"])), float(max(col_info["FREQ_TOP_PACK"])), float(min(col_info["FREQ_TOP_PACK"])))

    submitted = st.form_submit_button("Predict")

if submitted:
    # Step 1: Raw input into DataFrame
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
        "TOP_PACK": TOP_PACK,
        "FREQ_TOP_PACK": FREQ_TOP_PACK
    }])

    # Step 2: Feature Engineering & Encoding

    # --- Frequency Encoding (from col_info, based on full dataset) ---
    df['REGION_FE'] = df['REGION'].map(col_info["REGION_FE"]).fillna(0)
    df['TOP_PACK_FE'] = df['TOP_PACK'].map(col_info["TOP_PACK_FE"]).fillna(0)

    # --- Normalization of REGION_FE & TOP_PACK_FE ---
    df[['REGION_FE']] = MinMaxScaler().fit_transform(np.array(df['REGION_FE']).reshape(-1, 1))
    df[['TOP_PACK_FE']] = MinMaxScaler().fit_transform(np.array(df['TOP_PACK_FE']).reshape(-1, 1))

    # --- Ordinal Encoding for TENURE ---
    tenure_order = ['A < 1 month', 'B 1-3 month', 'C 3-6 month', 'D 6-9 month',
                    'E 9-12 month', 'F 12-15 month', 'G 15-18 month', 'H 18-21 month',
                    'I 21-24 month', 'J 24 month', 'K > 24 month']
    df['TENURE_OE'] = df['TENURE'].astype(pd.CategoricalDtype(categories=tenure_order, ordered=True)).cat.codes

    # --- Drop raw categorical fields ---
    df.drop(columns=['REGION', 'TENURE', 'TOP_PACK'], inplace=True)

    # Step 3: Scale Numerical Features
    num_cols = [
        'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
        'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
        'REGULARITY', 'FREQ_TOP_PACK'
    ]
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    # Step 4: Predict & Display
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.success("✅ Churn" if prediction == 1 else "❌ Not Churn")
    st.info(f"📈 Churn Probability: {prob:.2%}")
