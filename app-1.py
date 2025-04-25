import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and preprocessing objects
model = joblib.load("logistic_regression_model.joblib")

feature_order = joblib.load("feature_names.joblib")  # list of 28 column names

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



# URL to the dataset
url = 'https://www.dropbox.com/scl/fi/nyxsztvzq6391uof9gnlh/Expresso_churn_dataset.csv?rlkey=reo343zfzvvt8762ttapsirdd&e=1&st=xxagtwkr&raw=1'

# Load dataset
df1 = pd.read_csv(url)

# Clean column names by replacing spaces with underscores
df1.columns = df1.columns.str.replace(' ', '_')

# Define the columns for different types of encoding or selection
freq_cols = ['REGION', 'TOP_PACK']  # Frequency encoding or value mapping
onehot_cols = ['TENURE']  # One-hot encoding (mapped labels)
scale_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY']  # Scale/normalize

# Extract unique values for frequency encoding and one-hot encoding columns
freq_uniques = {col: df1[col].dropna().unique().tolist() for col in freq_cols}
onehot_uniques = {col: df1[col].dropna().unique().tolist() for col in onehot_cols}
scale_uniques = {col: df1[col].dropna().unique().tolist() for col in scale_cols}

# Output to check what’s extracted
st.write("Frequency Encoding Options:", freq_uniques)
st.write("One-Hot Encoding Options:", onehot_uniques)
st.write("Scaling Options:", scale_uniques)

# Now you can integrate this into your Streamlit app
# Select options for frequency encoded columns (e.g., region, top pack)
region = st.selectbox("Select Region", freq_uniques['REGION'])
top_pack = st.selectbox("Select Top Pack", freq_uniques['TOP_PACK'])

# Select options for one-hot encoded columns (e.g., tenure)
tenure = st.selectbox("Select Tenure", onehot_uniques['TENURE'])

# Accept numerical inputs for scaled features
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

# Create a button for prediction
if st.button("Predict Churn"):
    # Prepare the data dictionary for prediction
    input_data = {
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
        'REGULARITY': regularity,
        # Map frequency and one-hot encoded columns
        'REGION': region,
        'TOP_PACK': top_pack
    }

    # Convert to DataFrame for model input
    input_df = pd.DataFrame([input_data])

    

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
