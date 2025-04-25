import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing objects
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_order = joblib.load("feature_names.joblib")
col_info = joblib.load("col_info1.joblib")  # contains feature UI data

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
#Prediction
if st.button("Predict Churn"):
    # Step 1: Build raw input dict
    input_data = {
        'REGION': region,
        'TOP_PACK': top_pack,
        'TENURE': tenure,
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'REGULARITY': regularity
    }
    raw_df = pd.DataFrame([input_data])

    # Step 2: Apply frequency encoding
    for col in ['REGION', 'TOP_PACK']:
        freqs = df1[col].value_counts(normalize=True).to_dict()
        raw_df[col] = raw_df[col].map(freqs)

    # Step 3: Apply one-hot encoding for TENURE
    for key in tenure_order.keys():
        raw_df[f'TENURE_{key}'] = 1 if tenure == key else 0
    raw_df.drop(columns=['TENURE'], inplace=True)

    # Step 4: Scale numerical columns
    scale_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE',
                  'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY']

    scaler = joblib.load("scaler.joblib")
    scaled_values = scaler.transform(raw_df[scale_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=scale_cols)

    # Step 5: Merge all into final input
    others = raw_df.drop(columns=scale_cols)  # region/top_pack (freq) + TENURE_*
    input_df = pd.concat([others, scaled_df], axis=1)

    # Step 6: Align column order to match training
    input_df = input_df.reindex(columns=feature_order, fill_value=0)

    # Step 7: Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display results
    st.subheader("📊 Prediction Result")
    st.write("Churn Prediction:", "Yes ✅" if prediction == 1 else "No ❌")
    st.write(f"Churn Probability: {probability:.2%}")

