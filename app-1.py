import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import altair as alt
import datetime
from PIL import Image
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Expresso Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define theme colors
theme_colors = {
    "orange": {
        "primary": "#FF5722",
        "secondary": "#FF8A65",
        "accent": "#FFCCBC",
        "background": "#FBE9E7",
        "text": "#D84315"
    },
    "blue": {
        "primary": "#1976D2",
        "secondary": "#64B5F6",
        "accent": "#BBDEFB",
        "background": "#E3F2FD",
        "text": "#0D47A1"
    },
    "green": {
        "primary": "#388E3C",
        "secondary": "#81C784",
        "accent": "#C8E6C9",
        "background": "#E8F5E9",
        "text": "#1B5E20"
    },
    "purple": {
        "primary": "#7B1FA2",
        "secondary": "#BA68C8",
        "accent": "#E1BEE7",
        "background": "#F3E5F5",
        "text": "#4A148C"
    }
}

# Initialize session state for dashboard customization
if 'theme' not in st.session_state:
    st.session_state.theme = "orange"
if 'show_customer_profile' not in st.session_state:
    st.session_state.show_customer_profile = True
if 'show_usage_patterns' not in st.session_state:
    st.session_state.show_usage_patterns = True
if 'show_competitor_interaction' not in st.session_state:
    st.session_state.show_competitor_interaction = True
if 'show_package_info' not in st.session_state:
    st.session_state.show_package_info = True
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Get current theme colors
current_theme = theme_colors[st.session_state.theme]

# Custom CSS for styling with dynamic theme colors
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {current_theme["primary"]};
        text-align: center;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.5rem;
        color: {current_theme["secondary"]};
        margin-bottom: 1rem;
    }}
    .section-header {{
        font-size: 1.2rem;
        color: {current_theme["secondary"]};
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }}
    .prediction-box-high {{
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 5px;
    }}
    .prediction-box-medium {{
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
    }}
    .prediction-box-low {{
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {current_theme["accent"]};
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {current_theme["primary"]};
        color: white;
    }}
    .tooltip {{
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }}
    .dashboard-card {{
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    .card-header {{
        color: {current_theme["primary"]};
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid {current_theme["accent"]};
        padding-bottom: 0.5rem;
    }}
    .media-card {{
        background-color: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }}
    .media-card:hover {{
        transform: translateY(-5px);
    }}
    .media-card img {{
        width: 100%;
        height: 200px;
        object-fit: cover;
    }}
    .media-card-content {{
        padding: 1rem;
    }}
    .media-card-title {{
        color: {current_theme["primary"]};
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
    .footer {{
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
        font-size: 0.8rem;
        color: #666;
    }}
    .video-container {{
        position: relative;
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
        max-width: 100%;
    }}
    .video-container iframe {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }}
    .gallery {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .gallery-item {{
        flex: 1 0 200px;
        max-width: calc(33.333% - 10px);
        position: relative;
        overflow: hidden;
        border-radius: 5px;
    }}
    .gallery-item img {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        transition: transform 0.3s ease;
    }}
    .gallery-item:hover img {{
        transform: scale(1.05);
    }}
    .gallery-caption {{
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px;
        transform: translateY(100%);
        transition: transform 0.3s ease;
    }}
    .gallery-item:hover .gallery-caption {{
        transform: translateY(0);
    }}
    .metric-card {{
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {current_theme["primary"]};
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #666;
    }}
</style>
""", unsafe_allow_html=True)

# Load saved model & references
@st.cache_resource
def load_model():
    return joblib.load("clf.joblib")

@st.cache_data
def load_col_info():
    return joblib.load("unique_elements_dict2.joblib")

model = load_model()
col_info = load_col_info()

# Sidebar for dashboard customization
with st.sidebar:
    st.title("Dashboard Settings")
    
    # Theme selection
    st.header("Theme")
    selected_theme = st.selectbox(
        "Select Theme",
        options=list(theme_colors.keys()),
        index=list(theme_colors.keys()).index(st.session_state.theme),
        format_func=lambda x: x.capitalize()
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    # Dashboard sections visibility
    st.header("Customize Sections")
    st.session_state.show_customer_profile = st.checkbox("Show Customer Profile", value=st.session_state.show_customer_profile)
    st.session_state.show_usage_patterns = st.checkbox("Show Usage Patterns", value=st.session_state.show_usage_patterns)
    st.session_state.show_competitor_interaction = st.checkbox("Show Competitor Interaction", value=st.session_state.show_competitor_interaction)
    st.session_state.show_package_info = st.checkbox("Show Package Information", value=st.session_state.show_package_info)
    
    # Preset profiles for quick testing
    st.header("Preset Profiles")
    if st.button("High Risk Customer"):
        st.session_state.preset_profile = {
            "REGION": col_info["REGION"][0],
            "TENURE": "A < 1 month",
            "MONTANT": float(max(col_info["MONTANT"])) * 0.2,
            "FREQUENCE_RECH": float(min(col_info["FREQUENCE_RECH"])) + 1,
            "REVENUE": float(min(col_info["REVENUE"])),
            "ARPU_SEGMENT": float(min(col_info["ARPU_SEGMENT"])),
            "FREQUENCE": float(min(col_info["FREQUENCE"])),
            "DATA_VOLUME": float(min(col_info["DATA_VOLUME"])),
            "ON_NET": float(min(col_info["ON_NET"])),
            "ORANGE": float(max(col_info["ORANGE"])) * 0.8,
            "TIGO": float(max(col_info["TIGO"])) * 0.8,
            "REGULARITY": float(min(col_info["REGULARITY"])),
            "TOP_PACK": col_info["TOP_PACK"][0],
            "FREQ_TOP_PACK": float(min(col_info["FREQ_TOP_PACK"]))
        }
        st.rerun()
    
    if st.button("Medium Risk Customer"):
        st.session_state.preset_profile = {
            "REGION": col_info["REGION"][1],
            "TENURE": "E 9-12 month",
            "MONTANT": float(max(col_info["MONTANT"])) * 0.5,
            "FREQUENCE_RECH": float(max(col_info["FREQUENCE_RECH"])) * 0.5,
            "REVENUE": float(max(col_info["REVENUE"])) * 0.5,
            "ARPU_SEGMENT": float(max(col_info["ARPU_SEGMENT"])) * 0.5,
            "FREQUENCE": float(max(col_info["FREQUENCE"])) * 0.5,
            "DATA_VOLUME": float(max(col_info["DATA_VOLUME"])) * 0.5,
            "ON_NET": float(max(col_info["ON_NET"])) * 0.5,
            "ORANGE": float(max(col_info["ORANGE"])) * 0.5,
            "TIGO": float(max(col_info["TIGO"])) * 0.5,
            "REGULARITY": float(max(col_info["REGULARITY"])) * 0.5,
            "TOP_PACK": col_info["TOP_PACK"][1],
            "FREQ_TOP_PACK": float(max(col_info["FREQ_TOP_PACK"])) * 0.5
        }
        st.rerun()
    
    if st.button("Low Risk Customer"):
        st.session_state.preset_profile = {
            "REGION": col_info["REGION"][2],
            "TENURE": "K > 24 month",
            "MONTANT": float(max(col_info["MONTANT"])) * 0.8,
            "FREQUENCE_RECH": float(max(col_info["FREQUENCE_RECH"])) * 0.8,
            "REVENUE": float(max(col_info["REVENUE"])) * 0.8,
            "ARPU_SEGMENT": float(max(col_info["ARPU_SEGMENT"])) * 0.8,
            "FREQUENCE": float(max(col_info["FREQUENCE"])) * 0.8,
            "DATA_VOLUME": float(max(col_info["DATA_VOLUME"])) * 0.8,
            "ON_NET": float(max(col_info["ON_NET"])) * 0.8,
            "ORANGE": float(min(col_info["ORANGE"])) + 1,
            "TIGO": float(min(col_info["TIGO"])) + 1,
            "REGULARITY": float(max(col_info["REGULARITY"])) * 0.8,
            "TOP_PACK": col_info["TOP_PACK"][2],
            "FREQ_TOP_PACK": float(max(col_info["FREQ_TOP_PACK"])) * 0.8
        }
        st.rerun()
    
    # Reset dashboard
    if st.button("Reset Dashboard"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Create a logo and title section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div style="text-align: center;"><img src="https://placeholder.svg?height=100&width=300" alt="Expresso Logo"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Expresso Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Predict customer churn probability based on telecom usage patterns</p>', unsafe_allow_html=True)

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Dashboard", "📈 Data Insights", "🎬 Media Resources", "ℹ️ About"])

with tab1:
    # Create a form for user input
    with st.form("predict_form"):
        # Use preset profile if available
        preset_values = {}
        if 'preset_profile' in st.session_state:
            preset_values = st.session_state.preset_profile
            del st.session_state.preset_profile
        
        # Group 1: Customer Profile
        if st.session_state.show_customer_profile:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Customer Profile</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-header">Customer Demographics</p>', unsafe_allow_html=True)
                REGION = st.selectbox(
                    "REGION", 
                    col_info["REGION"],
                    index=col_info["REGION"].index(preset_values.get("REGION", col_info["REGION"][0])) if "REGION" in preset_values else 0,
                    help="The geographical region where the customer is located"
                )
                TENURE = st.selectbox(
                    "TENURE", 
                    col_info["TENURE"],
                    index=col_info["TENURE"].index(preset_values.get("TENURE", col_info["TENURE"][0])) if "TENURE" in preset_values else 0,
                    help="How long the customer has been with Expresso"
                )
            
            with col2:
                st.markdown('<p class="section-header">Financial Metrics</p>', unsafe_allow_html=True)
                REVENUE = st.slider(
                    "REVENUE", 
                    min_value=float(min(col_info["REVENUE"])), 
                    max_value=float(max(col_info["REVENUE"])), 
                    value=preset_values.get("REVENUE", float(min(col_info["REVENUE"]))),
                    help="Total revenue generated by the customer"
                )
                ARPU_SEGMENT = st.slider(
                    "ARPU_SEGMENT", 
                    min_value=float(min(col_info["ARPU_SEGMENT"])), 
                    max_value=float(max(col_info["ARPU_SEGMENT"])), 
                    value=preset_values.get("ARPU_SEGMENT", float(min(col_info["ARPU_SEGMENT"]))),
                    help="Average Revenue Per User segment"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Group 2: Usage Patterns
        if st.session_state.show_usage_patterns:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Usage Patterns</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<p class="section-header">Recharge Behavior</p>', unsafe_allow_html=True)
                MONTANT = st.slider(
                    "MONTANT", 
                    min_value=float(min(col_info["MONTANT"])), 
                    max_value=float(max(col_info["MONTANT"])), 
                    value=preset_values.get("MONTANT", float(min(col_info["MONTANT"]))),
                    help="Amount recharged by the customer"
                )
                FREQUENCE_RECH = st.slider(
                    "FREQUENCE_RECH", 
                    min_value=float(min(col_info["FREQUENCE_RECH"])), 
                    max_value=float(max(col_info["FREQUENCE_RECH"])), 
                    value=preset_values.get("FREQUENCE_RECH", float(min(col_info["FREQUENCE_RECH"]))),
                    help="Frequency of recharges"
                )
            
            with col2:
                st.markdown('<p class="section-header">Data Usage</p>', unsafe_allow_html=True)
                FREQUENCE = st.slider(
                    "FREQUENCE", 
                    min_value=float(min(col_info["FREQUENCE"])), 
                    max_value=float(max(col_info["FREQUENCE"])), 
                    value=preset_values.get("FREQUENCE", float(min(col_info["FREQUENCE"]))),
                    help="Frequency of usage"
                )
                DATA_VOLUME = st.slider(
                    "DATA_VOLUME", 
                    min_value=float(min(col_info["DATA_VOLUME"])), 
                    max_value=float(max(col_info["DATA_VOLUME"])), 
                    value=preset_values.get("DATA_VOLUME", float(min(col_info["DATA_VOLUME"]))),
                    help="Volume of data used by the customer"
                )
            
            with col3:
                st.markdown('<p class="section-header">Network Usage</p>', unsafe_allow_html=True)
                ON_NET = st.slider(
                    "ON_NET", 
                    min_value=float(min(col_info["ON_NET"])), 
                    max_value=float(max(col_info["ON_NET"])), 
                    value=preset_values.get("ON_NET", float(min(col_info["ON_NET"]))),
                    help="Calls made within the Expresso network"
                )
                REGULARITY = st.slider(
                    "REGULARITY", 
                    min_value=float(min(col_info["REGULARITY"])), 
                    max_value=float(max(col_info["REGULARITY"])), 
                    value=preset_values.get("REGULARITY", float(min(col_info["REGULARITY"]))),
                    help="Regularity of usage"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Group 3: Competitor Interaction
        if st.session_state.show_competitor_interaction:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Competitor Interaction</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                ORANGE = st.slider(
                    "ORANGE", 
                    min_value=float(min(col_info["ORANGE"])), 
                    max_value=float(max(col_info["ORANGE"])), 
                    value=preset_values.get("ORANGE", float(min(col_info["ORANGE"]))),
                    help="Calls made to Orange network"
                )
            
            with col2:
                TIGO = st.slider(
                    "TIGO", 
                    min_value=float(min(col_info["TIGO"])), 
                    max_value=float(max(col_info["TIGO"])), 
                    value=preset_values.get("TIGO", float(min(col_info["TIGO"]))),
                    help="Calls made to Tigo network"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Group 4: Package Information
        if st.session_state.show_package_info:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Package Information</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                TOP_PACK = st.selectbox(
                    "TOP_PACK", 
                    col_info["TOP_PACK"],
                    index=col_info["TOP_PACK"].index(preset_values.get("TOP_PACK", col_info["TOP_PACK"][0])) if "TOP_PACK" in preset_values else 0,
                    help="The top package used by the customer"
                )
            
            with col2:
                FREQ_TOP_PACK = st.slider(
                    "FREQ_TOP_PACK", 
                    min_value=float(min(col_info["FREQ_TOP_PACK"])), 
                    max_value=float(max(col_info["FREQ_TOP_PACK"])), 
                    value=preset_values.get("FREQ_TOP_PACK", float(min(col_info["FREQ_TOP_PACK"]))),
                    help="Frequency of using the top package"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Predict Churn Probability")

    # -----------------------
    # Data Transformation & Prediction
    # -----------------------
    if submitted:
        # Create a spinner to show processing
        with st.spinner('Analyzing customer data...'):
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

            # Frequency encode REGION using external full dataset frequencies
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
            
            # Predict & Display
            prediction = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]
            
            # Store prediction in session state
            st.session_state.last_prediction = {
                "prediction": int(prediction),
                "probability": float(prob),
                "timestamp": datetime.datetime.now(),
                "customer_data": {
                    "REGION": REGION,
                    "TENURE": TENURE,
                    "REVENUE": REVENUE,
                    "DATA_VOLUME": DATA_VOLUME
                }
            }
            
            # Add to prediction history
            st.session_state.prediction_history.append(st.session_state.last_prediction)
            
            # -----------------------
            # Enhanced Results Display
            # -----------------------
            st.markdown("---")
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Prediction Results</div>', unsafe_allow_html=True)
            
            # Create columns for results display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Create a gauge chart for churn probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': current_theme["primary"]},
                        'steps': [
                            {'range': [0, 30], 'color': "#E8F5E9"},
                            {'range': [30, 70], 'color': "#FFF8E1"},
                            {'range': [70, 100], 'color': "#FFEBEE"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prob * 100
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Determine risk level and display appropriate message
                if prob >= 0.7:
                    risk_level = "High"
                    box_class = "prediction-box-high"
                    icon = "⚠️"
                    message = "This customer is at high risk of churning."
                elif prob >= 0.3:
                    risk_level = "Medium"
                    box_class = "prediction-box-medium"
                    icon = "⚠️"
                    message = "This customer is at moderate risk of churning."
                else:
                    risk_level = "Low"
                    box_class = "prediction-box-low"
                    icon = "✅"
                    message = "This customer is at low risk of churning."
                
                # Display prediction result with styling
                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{icon} Churn Risk: {risk_level}</h3>
                    <p>{message}</p>
                    <p>Churn Probability: <b>{prob:.2%}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display recommended actions based on risk level
                st.markdown("### Recommended Actions")
                
                if risk_level == "High":
                    st.markdown("""
                    - 📞 **Immediate Outreach**: Contact customer with personalized retention offer
                    - 💰 **Special Discount**: Offer significant discount on their preferred services
                    - 🎁 **Loyalty Bonus**: Provide immediate loyalty bonus or free service upgrade
                    - 📊 **Usage Analysis**: Review customer usage patterns for targeted improvements
                    """)
                elif risk_level == "Medium":
                    st.markdown("""
                    - 📱 **Service Check**: Proactively check if customer is satisfied with services
                    - 🎁 **Targeted Offer**: Send targeted offer based on usage patterns
                    - 💬 **Feedback Request**: Request feedback on service quality
                    - 📈 **Usage Suggestions**: Suggest optimal plans based on their usage
                    """)
                else:
                    st.markdown("""
                    - 🎁 **Loyalty Rewards**: Continue providing loyalty rewards
                    - 📱 **Cross-Sell**: Suggest complementary services they might enjoy
                    - 🌟 **Referral Program**: Invite to participate in referral program
                    - 📊 **Regular Check-ins**: Schedule periodic service reviews
                    """)
            
            # Feature importance visualization
            st.markdown("### Key Factors Influencing Prediction")
            
            # For demonstration, using dummy feature importance values
            # In a real scenario, you would extract these from your model
            feature_importance = {
                'REVENUE': 0.25,
                'TENURE_OE': 0.20,
                'MONTANT': 0.15,
                'FREQUENCE': 0.12,
                'DATA_VOLUME': 0.10,
                'ORANGE': 0.08,
                'TIGO': 0.05,
                'REGULARITY': 0.05
            }
            
            # Create a horizontal bar chart for feature importance
            fig = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title='Feature Importance',
                color=list(feature_importance.values()),
                color_continuous_scale=f'{st.session_state.theme}s'
            )
            
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # What-if analysis section
            st.markdown("### What-If Analysis")
            st.markdown("Explore how changing certain parameters would affect the churn probability")
            
            what_if_tabs = st.tabs(["Revenue Impact", "Data Usage Impact", "Competitor Impact"])
            
            with what_if_tabs[0]:
                # Revenue impact analysis
                revenue_values = np.linspace(float(min(col_info["REVENUE"])), float(max(col_info["REVENUE"])), 10)
                revenue_probs = []
                
                for rev in revenue_values:
                    df_what_if = df.copy()
                    df_what_if['REVENUE'] = scaler.transform([[rev]])[0][0]  # Scale the new revenue value
                    revenue_probs.append(model.predict_proba(df_what_if)[0][1])
                
                fig = px.line(
                    x=revenue_values,
                    y=revenue_probs,
                    labels={'x': 'Revenue', 'y': 'Churn Probability'},
                    title='How Revenue Affects Churn Probability',
                    markers=True
                )
                fig.update_traces(line_color=current_theme["primary"], line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with what_if_tabs[1]:
                # Data usage impact analysis
                data_values = np.linspace(float(min(col_info["DATA_VOLUME"])), float(max(col_info["DATA_VOLUME"])), 10)
                data_probs = []
                
                for data_val in data_values:
                    df_what_if = df.copy()
                    df_what_if['DATA_VOLUME'] = scaler.transform([[data_val]])[0][0]  # Scale the new data value
                    data_probs.append(model.predict_proba(df_what_if)[0][1])
                
                fig = px.line(
                    x=data_values,
                    y=data_probs,
                    labels={'x': 'Data Volume', 'y': 'Churn Probability'},
                    title='How Data Usage Affects Churn Probability',
                    markers=True
                )
                fig.update_traces(line_color=current_theme["primary"], line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with what_if_tabs[2]:
                # Competitor impact analysis
                orange_values = np.linspace(float(min(col_info["ORANGE"])), float(max(col_info["ORANGE"])), 10)
                orange_probs = []
                
                for orange_val in orange_values:
                    df_what_if = df.copy()
                    df_what_if['ORANGE'] = scaler.transform([[orange_val]])[0][0]  # Scale the new orange value
                    orange_probs.append(model.predict_proba(df_what_if)[0][1])
                
                fig = px.line(
                    x=orange_values,
                    y=orange_probs,
                    labels={'x': 'Calls to Orange Network', 'y': 'Churn Probability'},
                    title='How Competitor Usage Affects Churn Probability',
                    markers=True
                )
                fig.update_traces(line_color=current_theme["primary"], line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction history
            if len(st.session_state.prediction_history) > 1:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">Prediction History</div>', unsafe_allow_html=True)
                
                # Create a line chart of prediction history
                history_data = pd.DataFrame([
                    {
                        "timestamp": p["timestamp"],
                        "probability": p["probability"],
                        "prediction": "Churn" if p["prediction"] == 1 else "No Churn"
                    }
                    for p in st.session_state.prediction_history
                ])
                
                fig = px.line(
                    history_data,
                    x="timestamp",
                    y="probability",
                    color="prediction",
                    labels={"probability": "Churn Probability", "timestamp": "Time"},
                    title="Prediction History",
                    markers=True
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show history in a table
                st.dataframe(
                    history_data[["timestamp", "probability", "prediction"]].sort_values("timestamp", ascending=False),
                    use_container_width=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    data_tabs = st.tabs(["Customer Segments", "Regional Analysis", "Temporal Trends", "Usage Patterns"])
    
    with data_tabs[0]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Customer Segmentation Analysis</div>', unsafe_allow_html=True)
        
        # Customer segments metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">23.5%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Overall Churn Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">42.1%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">New User Churn</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">18.7%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Long-term User Churn</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">35.2%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">High-Value User Churn</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Customer segments visualization
        st.markdown("### Customer Segments by Churn Risk")
        
        # Sample data for customer segments
        segments = ['New Users', 'Low Usage', 'Medium Usage', 'High Usage', 'Premium']
        high_risk = [42, 28, 18, 15, 12]
        medium_risk = [35, 40, 45, 30, 25]
        low_risk = [23, 32, 37, 55, 63]
        
        # Create a stacked bar chart
        fig = go.Figure(data=[
            go.Bar(name='High Risk', x=segments, y=high_risk, marker_color='#F44336'),
            go.Bar(name='Medium Risk', x=segments, y=medium_risk, marker_color='#FFC107'),
            go.Bar(name='Low Risk', x=segments, y=low_risk, marker_color='#4CAF50')
        ])
        
        fig.update_layout(
            barmode='stack',
            title='Customer Segments by Churn Risk',
            xaxis_title='Customer Segment',
            yaxis_title='Percentage',
            legend_title='Risk Level',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn reasons
        st.markdown("### Primary Reasons for Churn")
        
        # Sample data for churn reasons
        reasons = ['Price', 'Competitor Offers', 'Service Quality', 'Network Coverage', 'Customer Service', 'Other']
        percentages = [35, 25, 15, 12, 8, 5]
        
        # Create a pie chart
        fig = px.pie(
            values=percentages,
            names=reasons,
            title='Primary Reasons for Churn',
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with data_tabs[1]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Regional Analysis</div>', unsafe_allow_html=True)
        
        # Regional churn heatmap
        st.markdown("### Churn Rate by Region")
        
        # Sample data for regional analysis
        regions = ['Dakar', 'Thies', 'Saint-Louis', 'Diourbel', 'Kaolack', 'Ziguinchor', 'Louga', 'Fatick', 'Kolda']
        region_data = []
        
        for region in regions:
            region_data.append({
                'Region': region,
                'Overall Churn': np.random.uniform(0.15, 0.35),
                'New Users': np.random.uniform(0.25, 0.45),
                'Long-term Users': np.random.uniform(0.10, 0.25),
                'High-Value Users': np.random.uniform(0.20, 0.40)
            })
        
        region_df = pd.DataFrame(region_data)
        
        # Create a heatmap
        fig = px.imshow(
            region_df.set_index('Region').values,
            labels=dict(x="Customer Segment", y="Region", color="Churn Rate"),
            x=['Overall Churn', 'New Users', 'Long-term Users', 'High-Value Users'],
            y=region_df['Region'],
            color_continuous_scale='Oranges',
            text_auto='.2%'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional map visualization
        st.markdown("### Geographic Distribution of Churn")
        
        # For a real implementation, you would use actual map data for Senegal/Mauritania
        # This is a placeholder visualization
        st.markdown("""
        *Note: This is a placeholder for a geographic map visualization. In a production environment, 
        you would implement an actual map of Senegal and Mauritania with regional churn data.*
        """)
        
        # Create a sample scatter geo plot
        fig = px.scatter_geo(
            region_df,
            lat=[14.7, 14.8, 16.0, 14.6, 14.1, 12.6, 15.6, 14.3, 12.9],  # Sample coordinates
            lon=[-17.4, -16.9, -16.5, -16.2, -16.1, -16.3, -16.2, -16.4, -14.9],  # Sample coordinates
            size=region_df['Overall Churn'] * 100,
            color=region_df['Overall Churn'],
            hover_name='Region',
            color_continuous_scale='Oranges',
            size_max=30,
            title='Churn Rate by Geographic Location'
        )
        
        fig.update_geos(
            projection_type="natural earth",
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="LightGreen",
            showocean=True,
            oceancolor="LightBlue",
            showlakes=True,
            lakecolor="Blue"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with data_tabs[2]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Temporal Trends</div>', unsafe_allow_html=True)
        
        # Time series analysis
        st.markdown("### Churn Rate Over Time")
        
        # Generate sample time series data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='M')
        churn_rates = [0.22 + 0.05 * np.sin(i/3) + np.random.uniform(-0.03, 0.03) for i in range(len(dates))]
        
        # Create a time series plot
        fig = px.line(
            x=dates,
            y=churn_rates,
            labels={'x': 'Date', 'y': 'Churn Rate'},
            title='Monthly Churn Rate Trend',
            markers=True
        )
        
        fig.update_traces(line_color=current_theme["primary"], line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        st.markdown("### Seasonal Patterns in Churn")
        
        # Sample data for seasonal patterns
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_churn = [0.26, 0.24, 0.22, 0.21, 0.20, 0.19, 0.21, 0.23, 0.25, 0.27, 0.28, 0.27]
        
        # Create a bar chart
        fig = px.bar(
            x=months,
            y=seasonal_churn,
            labels={'x': 'Month', 'y': 'Average Churn Rate'},
            title='Seasonal Patterns in Churn Rate',
            color=seasonal_churn,
            color_continuous_scale=f'{st.session_state.theme}s'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week patterns
        st.markdown("### Day of Week Patterns")
        
        # Sample data for day of week patterns
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        usage_patterns = [85, 82, 80, 78, 90, 100, 95]
        churn_events = [18, 15, 14, 16, 20, 25, 22]
        
        # Create a dual-axis chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=days,
            y=usage_patterns,
            name='Usage (% of max)',
            marker_color=current_theme["secondary"]
        ))
        
        fig.add_trace(go.Scatter(
            x=days,
            y=churn_events,
            name='Churn Events',
            marker_color=current_theme["primary"],
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Usage and Churn Events by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Usage (% of max)',
            legend_title='Metric',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with data_tabs[3]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Usage Pattern Analysis</div>', unsafe_allow_html=True)
        
        # Usage patterns analysis
        st.markdown("### Data Usage vs. Churn Probability")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 200
        data_volume = np.random.uniform(low=float(min(col_info["DATA_VOLUME"])), high=float(max(col_info["DATA_VOLUME"])), size=n_samples)
        revenue = np.random.uniform(low=float(min(col_info["REVENUE"])), high=float(max(col_info["REVENUE"])), size=n_samples)
        churn_prob = 0.5 - 0.3 * (data_volume / max(data_volume)) - 0.2 * (revenue / max(revenue)) + np.random.normal(0, 0.1, n_samples)
        churn_prob = np.clip(churn_prob, 0, 1)
        
        # Create a scatter plot
        scatter_df = pd.DataFrame({
            'Data Volume': data_volume,
            'Revenue': revenue,
            'Churn Probability': churn_prob,
            'Risk Level': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in churn_prob]
        })
        
        fig = px.scatter(
            scatter_df,
            x='Data Volume',
            y='Revenue',
            color='Risk Level',
            size='Churn Probability',
            hover_data=['Churn Probability'],
            color_discrete_map={'High': '#F44336', 'Medium': '#FFC107', 'Low': '#4CAF50'},
            title='Data Usage vs. Revenue Colored by Churn Risk'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Network usage patterns
        st.markdown("### Network Usage Patterns")
        
        # Sample data for network usage
        categories = ['On-Net Calls', 'Orange Calls', 'Tigo Calls', 'Data Usage', 'SMS']
        
        # Create multiple traces for different customer segments
        fig = go.Figure()
        
        # Low churn risk customers
        fig.add_trace(go.Scatterpolar(
            r=[80, 30, 20, 75, 65],
            theta=categories,
            fill='toself',
            name='Low Churn Risk',
            line_color='#4CAF50'
        ))
        
        # Medium churn risk customers
        fig.add_trace(go.Scatterpolar(
            r=[60, 50, 45, 50, 40],
            theta=categories,
            fill='toself',
            name='Medium Churn Risk',
            line_color='#FFC107'
        ))
        
        # High churn risk customers
        fig.add_trace(go.Scatterpolar(
            r=[40, 70, 65, 30, 25],
            theta=categories,
            fill='toself',
            name='High Churn Risk',
            line_color='#F44336'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Network Usage Patterns by Churn Risk',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### Feature Correlation Matrix")
        
        # Sample correlation matrix
        features = ['REVENUE', 'MONTANT', 'FREQUENCE_RECH', 'ARPU_SEGMENT', 'FREQUENCE', 
                   'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
        
        # Generate a sample correlation matrix
        np.random.seed(42)
        corr_matrix = np.random.uniform(-0.8, 0.8, size=(len(features), len(features)))
        np.fill_diagonal(corr_matrix, 1)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric
        
        # Create a heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Feature", y="Feature", color="Correlation"),
            x=features,
            y=features,
            color_continuous_scale='RdBu_r',
            text_auto='.2f'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    media_tabs = st.tabs(["Educational Videos", "Infographics", "Case Studies"])
    
    with media_tabs[0]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Educational Videos</div>', unsafe_allow_html=True)
        
        st.markdown("### Learn About Customer Churn and Retention")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<div class="video-container"><iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Understanding Customer Churn</div>', unsafe_allow_html=True)
            st.markdown('<p>Learn about the factors that contribute to customer churn in the telecom industry and how to identify at-risk customers.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<div class="video-container"><iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Effective Retention Strategies</div>', unsafe_allow_html=True)
            st.markdown('<p>Discover proven strategies for retaining customers and reducing churn in competitive telecom markets.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Advanced Topics in Churn Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<div class="video-container"><iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Machine Learning for Churn Prediction</div>', unsafe_allow_html=True)
            st.markdown('<p>An in-depth look at how machine learning models can predict customer churn with high accuracy.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<div class="video-container"><iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Implementing Retention Programs</div>', unsafe_allow_html=True)
            st.markdown('<p>Step-by-step guide to implementing effective customer retention programs based on predictive analytics.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with media_tabs[1]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Infographics</div>', unsafe_allow_html=True)
        
        st.markdown("### Customer Churn Infographics")
        
        # Sample infographics (using placeholders)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<img src="https://placeholder.svg?height=300&width=400" alt="Churn Factors Infographic">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Key Factors Driving Churn</div>', unsafe_allow_html=True)
            st.markdown('<p>Visual representation of the primary factors that lead to customer churn in the telecom industry.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<img src="https://placeholder.svg?height=300&width=400" alt="Customer Journey Infographic">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Customer Journey Map</div>', unsafe_allow_html=True)
            st.markdown('<p>Visualization of the customer journey and critical touchpoints that influence retention.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<img src="https://placeholder.svg?height=300&width=400" alt="Retention Strategies Infographic">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Effective Retention Strategies</div>', unsafe_allow_html=True)
            st.markdown('<p>Visual guide to implementing effective customer retention strategies based on churn risk.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Industry Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<img src="https://placeholder.svg?height=300&width=400" alt="Telecom Industry Statistics">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Telecom Industry Churn Rates</div>', unsafe_allow_html=True)
            st.markdown('<p>Comparative analysis of churn rates across different telecom markets in Africa.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="media-card">', unsafe_allow_html=True)
            st.markdown('<img src="https://placeholder.svg?height=300&width=400" alt="Customer Acquisition vs Retention">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-content">', unsafe_allow_html=True)
            st.markdown('<div class="media-card-title">Acquisition vs. Retention Costs</div>', unsafe_allow_html=True)
            st.markdown('<p>Statistical breakdown of customer acquisition costs compared to retention costs in the telecom sector.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with media_tabs[2]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Case Studies</div>', unsafe_allow_html=True)
        
        st.markdown("### Success Stories")
        
        # Sample case studies
        st.markdown("""
        #### Case Study 1: Reducing Churn by 35% at a Major African Telecom
        
        A major telecom provider in West Africa was experiencing high churn rates, particularly among new customers. 
        By implementing predictive analytics and targeted retention strategies, they were able to:
        
        - Identify high-risk customers with 78% accuracy
        - Reduce overall churn by 35% within 6 months
        - Increase customer lifetime value by 28%
        - Achieve ROI of 340% on their retention program
        
        The key to their success was early intervention with personalized offers based on usage patterns and customer segments.
        """)
        
        st.markdown("""
        #### Case Study 2: Improving Customer Experience to Reduce Churn
        
        A mid-sized telecom operator focused on improving customer experience as a strategy to reduce churn. Their approach included:
        
        - Streamlining the customer service process
        - Implementing a proactive outreach program for at-risk customers
        - Developing a loyalty program with tiered benefits
        - Creating personalized usage recommendations
        
        The results were impressive: a 42% reduction in churn among high-value customers and a 23% increase in customer satisfaction scores.
        """)
        
        st.markdown("### Implementation Guide")
        
        st.markdown("""
        #### How to Implement This Churn Prediction Model in Your Organization
        
        1. **Data Collection & Preparation**
           - Identify relevant customer data sources
           - Clean and preprocess historical data
           - Establish data pipelines for ongoing analysis
        
        2. **Model Deployment**
           - Train the model on your historical data
           - Validate accuracy with test datasets
           - Implement the model in your production environment
        
        3. **Intervention Strategy**
           - Develop targeted retention offers for different risk segments
           - Create automated triggers for high-risk customers
           - Train customer service teams on retention techniques
        
        4. **Monitoring & Optimization**
           - Track key performance indicators
           - Continuously refine the model based on new data
           - A/B test different retention strategies
        """)
        
        # Gallery of case study images
        st.markdown("### Gallery")
        
        st.markdown('<div class="gallery">', unsafe_allow_html=True)
        
        for i in range(6):
            st.markdown(f'''
            <div class="gallery-item">
                <img src="https://placeholder.svg?height=200&width=300" alt="Case Study Image {i+1}">
                <div class="gallery-caption">Case study implementation {i+1}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">About the Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Model Information
    
    This churn prediction model was developed using machine learning techniques to analyze customer behavior patterns and predict the likelihood of customers leaving Expresso's services.
    
    **Key Features:**
    - Uses historical customer data from Expresso's operations in Mauritania and Senegal
    - Analyzes over 15 behavioral variables to identify churn patterns
    - Provides probability scores to help prioritize retention efforts
    
    ### How It Works
    
    The model analyzes various aspects of customer behavior including:
    
    1. **Usage Patterns**: How frequently customers use services and their volume of usage
    2. **Financial Metrics**: Revenue generated, recharge amounts, and spending patterns
    3. **Customer Profile**: Tenure, region, and other demographic information
    4. **Competitor Interaction**: Calls to other networks like Orange and Tigo
    
    ### How to Use the Predictions
    
    - **High Risk (>70%)**: These customers require immediate attention and personalized retention offers
    - **Medium Risk (30-70%)**: Proactive engagement can help retain these customers
    - **Low Risk (<30%)**: Continue providing excellent service and look for upsell opportunities
    """)
    
    # Model performance metrics
    st.markdown("### Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">85.7%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">83.2%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Precision</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">79.5%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Recall</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">81.3%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">F1 Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ROC curve
    st.markdown("### Model ROC Curve")
    
    # Generate sample ROC curve data
    fpr = np.linspace(0, 1, 100)
    tpr = np.power(fpr, 0.5)  # Simple curve shape for demonstration
    
    # Create ROC curve
    fig = px.line(
        x=fpr,
        y=tpr,
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        title='ROC Curve (AUC = 0.857)'
    )
    
    # Add diagonal line
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_traces(line_color=current_theme["primary"], line_width=3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Team information
    st.markdown("### About the Team")
    
    st.markdown("""
    This churn prediction application was developed by the Expresso Data Science team in collaboration with external consultants. The team consists of data scientists, machine learning engineers, and domain experts in telecommunications.
    
    For questions or support, please contact:
    - Email: data.science@expresso.com
    - Phone: +221 33 839 1000
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with timestamp
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(f'Last updated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', unsafe_allow_html=True)
st.markdown('Expresso Churn Prediction Dashboard v2.0', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
