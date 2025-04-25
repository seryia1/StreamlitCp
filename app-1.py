import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load saved model & references
model = joblib.load("logistic_regression_model.joblib")
feature_names = joblib.load("feature_names.joblib")
col_info = joblib.load("col_info1.joblib")  # Contains options like TENURE, REGION, TOP_PACK

# -----------------------
# UI: User Input Form
# -----------------------
with st.form("predict_form"):
    st.title("📱 Churn Prediction (Expresso Users)")

    REGION = st.selectbox("REGION", col_info["REGION"])
    TOP_PACK = st.selectbox("TOP_PACK", col_info["TOP_PACK"])
    TENURE = st.selectbox("TENURE", col_info["TENURE"])

    MONTANT = st.slider("MONTANT", 0.0, 100000.0, 1000.0)
    FREQUENCE_RECH = st.slider("FREQUENCE_RECH", 0.0, 100.0, 5.0)
    REVENUE = st.slider("REVENUE", 0.0, 100000.0, 1000.0)
    ARPU_SEGMENT = st.slider("ARPU_SEGMENT", 0.0, 50000.0, 500.0)
    FREQUENCE = st.slider("FREQUENCE", 0.0, 100.0, 5.0)
    DATA_VOLUME = st.slider("DATA_VOLUME", 0.0, 100000.0, 500.0)
    ON_NET = st.slider("ON_NET", 0.0, 100000.0, 1000.0)
    ORANGE = st.slider("ORANGE", 0.0, 100000.0, 1000.0)
    TIGO = st.slider("TIGO", 0.0, 100000.0, 1000.0)
    REGULARITY = st.slider("REGULARITY", 0.0, 1.0, 0.5)
    FREQ_TOP_PACK = st.slider("FREQ_TOP_PACK", 0.0, 100.0, 5.0)

    submitted = st.form_submit_button("Predict")

# -----------------------
# Data Transformation
# -----------------------
if submitted:
    # 1. Raw input
    df = pd.DataFrame([{
        "REGION": REGION, "TOP_PACK": TOP_PACK, "TENURE": TENURE,
        "MONTANT": MONTANT, "FREQUENCE_RECH": FREQUENCE_RECH,
        "REVENUE": REVENUE, "ARPU_SEGMENT": ARPU_SEGMENT, "FREQUENCE": FREQUENCE,
        "DATA_VOLUME": DATA_VOLUME, "ON_NET": ON_NET, "ORANGE": ORANGE,
        "TIGO": TIGO, "REGULARITY": REGULARITY, "FREQ_TOP_PACK": FREQ_TOP_PACK
    }])

    # 2. Map TENURE
    tenure_order = {
        'A 0-3 month': 0, 'B 3-6 month': 1, 'C 6-9 month': 2, 'D 9-12 month': 3,
        'E 12-15 month': 4, 'F 15-18 month': 5, 'G 18-21 month': 6, 'H 21-24 month': 7,
        'I 18-21 month': 6, 'J 21-24 month': 7, 'K > 24 month': 8
    }
    df["TENURE"] = df["TENURE"].map(tenure_order)

    # 3. One-hot encode REGION (with NaN)
    df = pd.get_dummies(df, columns=["REGION"], prefix="REGION", dummy_na=True)

    # 4. Frequency encode TOP_PACK and normalize
    top_pack_freq = pd.Series(col_info["TOP_PACK"]).value_counts()
    df["TOP_PACK_FE"] = df["TOP_PACK"].map(top_pack_freq).fillna(0)
    df.drop(columns=["TOP_PACK"], inplace=True)

    minmax = MinMaxScaler()
    df["TOP_PACK_FE"] = minmax.fit_transform(df[["TOP_PACK_FE"]])

    # 5. Convert boolean (if any) to int
    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 6. Add missing one-hot REGION columns if needed
    for col in feature_names:
        if col.startswith("REGION_") and col not in df.columns:
            df[col] = 0

    # 7. Reorder columns and fill missing
    df = df.reindex(columns=feature_names, fill_value=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # -----------------------
    # Predict & Display
    # -----------------------
    prediction = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.success("✅ Churn" if prediction == 1 else "❌ Not Churn")
    st.info(f"📈 Churn Probability: {prob:.2%}")
