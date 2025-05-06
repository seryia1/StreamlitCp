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

    MONTANT = st.slider("MONTANT", min_value=float(min(col_info["MONTANT"])), max_value=float(max(col_info["MONTANT"])), value=float(min(col_info["MONTANT"])))
    FREQUENCE_RECH = st.slider("FREQUENCE_RECH", min_value=float(min(col_info["FREQUENCE_RECH"])), max_value=float(max(col_info["FREQUENCE_RECH"])), value=float(min(col_info["FREQUENCE_RECH"])))
    REVENUE = st.slider("REVENUE", min_value=float(min(col_info["REVENUE"])), max_value=float(max(col_info["REVENUE"])), value=float(min(col_info["REVENUE"])))
    ARPU_SEGMENT = st.slider("ARPU_SEGMENT", min_value=float(min(col_info["ARPU_SEGMENT"])), max_value=float(max(col_info["ARPU_SEGMENT"])), value=float(min(col_info["ARPU_SEGMENT"])))
    FREQUENCE = st.slider("FREQUENCE", min_value=float(min(col_info["FREQUENCE"])), max_value=float(max(col_info["FREQUENCE"])), value=float(min(col_info["FREQUENCE"])))
    DATA_VOLUME = st.slider("DATA_VOLUME", min_value=float(min(col_info["DATA_VOLUME"])), max_value=float(max(col_info["DATA_VOLUME"])), value=float(min(col_info["DATA_VOLUME"])))
    ON_NET = st.slider("ON_NET", min_value=float(min(col_info["ON_NET"])), max_value=float(max(col_info["ON_NET"])), value=float(min(col_info["ON_NET"])))
    ORANGE = st.slider("ORANGE", min_value=float(min(col_info["ORANGE"])), max_value=float(max(col_info["ORANGE"])), value=float(min(col_info["ORANGE"])))
    TIGO = st.slider("TIGO", min_value=float(min(col_info["TIGO"])), max_value=float(max(col_info["TIGO"])), value=float(min(col_info["TIGO"])))
    REGULARITY = st.slider("REGULARITY", min_value=float(min(col_info["REGULARITY"])), max_value=float(max(col_info["REGULARITY"])), value=float(min(col_info["REGULARITY"])))
    TOP_PACK=  st.selectbox("TOP_PACK", col_info["TOP_PACK"])
    FREQ_TOP_PACK = st.slider("FREQ_TOP_PACK", min_value=float(min(col_info["FREQ_TOP_PACK"])), max_value=float(max(col_info["FREQ_TOP_PACK"])), value=float(min(col_info["FREQ_TOP_PACK"])))
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
            "TOP_PACK": TOP_PACK,
            "FREQ_TOP_PACK": FREQ_TOP_PACK
        }])

        # Frequency encode REGION using external full dataset frequencies (col_info_region_freq must exist)
        region_freq = df['REGION'].value_counts(normalize=False)
        df['REGION_FE'] = df['REGION'].map(region_freq)

        # Normalize REGION_FE
        scaler_region = MinMaxScaler()
        df['REGION_FE'] = scaler_region.fit_transform(df[['REGION_FE']])

        # Ordinal encode TENURE
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
        # Drop original non-numeric columns
        df.drop(columns=['REGION', 'TENURE'], inplace=True)

        # Columns to scale (excluding target and already normalized/ordinal encoded ones)
        num_cols_to_scale = [
    'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
    'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK'
]

        scaler = StandardScaler()
        df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale]) 
        df
    # -----------------------
    # Predict & Display
    # -----------------------
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        st.success("✅ Churn" if prediction == 1 else "❌ Not Churn")
        st.info(f"📈 Churn Probability: {prob:.2%}")
