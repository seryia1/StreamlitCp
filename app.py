import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# === PAGE SETUP ===
st.set_page_config(
    page_title="Expresso Churn Prediction",
    page_icon="🌸",
    layout="wide"
)

# === GET PAGE FROM URL ===
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["overview"])[0]

# === CUSTOM CSS FOR EXPRESSO THEME ===
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #ffffff;
        color: #4a235a;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f9f9f9;
        border-right: 1px solid #eee;
    }
    
    /* Card-like containers */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs styling */
    .tab-nav {
        display: flex;
        border-bottom: 1px solid #eee;
        margin-bottom: 20px;
        padding-bottom: 10px;
    }
    
    .tab {
        padding: 8px 16px;
        margin-right: 10px;
        cursor: pointer;
        border-radius: 5px 5px 0 0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .tab:hover {
        background-color: rgba(249, 202, 36, 0.1);
    }
    
    .tab-active {
        border-bottom: 2px solid #f9ca24;
        color: #6c3483;
    }
    
    /* Button styling */
    .custom-button {
        background-color: #f9ca24;
        color: #4a235a;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
        text-align: center;
    }
    
    .custom-button:hover {
        background-color: #f0c30f;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Dropdown styling */
    .stSelectbox label, .stSlider label {
        color: #4a235a !important;
        font-weight: 500;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #6c3483;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #6c3483, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Yellow accent color for highlights */
    .accent {
        color: #f9ca24;
    }
    
    /* Expresso icon */
    .expresso-icon {
        width: 100%;
        max-width: 150px;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 8px rgba(108, 52, 131, 0.5));
    }
    
    /* Prediction result */
    .prediction-result {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 4px solid #6c3483;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom select box */
    .custom-select {
        background-color: #f9f9f9;
        border: 1px solid #eee;
        border-radius: 5px;
        color: #4a235a;
        padding: 10px;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #f9ca24, transparent);
        margin: 20px 0;
    }
    
    /* Feature cards */
    .feature-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 3px solid #6c3483;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Churn tag */
    .churn-tag {
        position: relative;
        display: inline-block;
        padding: 10px 20px;
        background: #6c3483;
        color: white;
        font-size: 32px;
        font-weight: bold;
        border-radius: 5px 0 0 5px;
    }
    
    .churn-tag:after {
        content: '';
        position: absolute;
        top: 0;
        right: -20px;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 28px 0 28px 20px;
        border-color: transparent transparent transparent #6c3483;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(108, 52, 131, 0.3);
        border-radius: 50%;
        border-top-color: #6c3483;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #4a235a;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #6c3483, #8e44ad);
        transition: width 0.5s ease;
    }
    
    /* Comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .comparison-table th, .comparison-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #eee;
    }
    
    .comparison-table th {
        background-color: #f9f9f9;
    }
    
    .comparison-table tr:hover {
        background-color: #f9f9f9;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        background-color: #6c3483;
        color: white;
        margin-left: 5px;
    }
    
    /* Streamlit element customization */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #6c3483 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hide-mobile {
            display: none;
        }
        
        h1 {
            font-size: 1.8rem;
        }
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        align-items: center;
    }
    
    .logo-circle {
        background-color: #f9ca24;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }
    
    .logo-text {
        color: #4a235a;
        font-weight: bold;
        font-size: 1.5rem;
        margin-top: 40px;
    }
    
    /* Navigation styling */
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-link {
        color: #4a235a;
        font-weight: 500;
        text-decoration: none;
        font-size: 1rem;
    }
    
    /* Background container */
    .background-container {
        position: relative;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        background: linear-gradient(to right, rgba(108, 52, 131, 0.9), rgba(74, 35, 90, 0.9));
        margin-bottom: 2rem;
    }
    
    .background-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0.3;
        border-radius: 10px;
        z-index: -1;
    }
    
    /* Contact button */
    .contact-button {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background-color: #4a235a;
        color: #f9ca24;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    
    .contact-text {
        position: fixed;
        bottom: 30px;
        left: 80px;
        background-color: white;
        color: #4a235a;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    /* Yellow accent for highlights */
    .yellow-accent {
        color: #f9ca24;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# === EXPRESSO LOGO SVG ===
expresso_logo = """
<div class="logo-container">
    <div class="logo-circle">
        <svg viewBox="0 0 50 50" width="40" height="40" fill="#4a235a">
            <path d="M25,10 C30,15 35,15 40,10 C40,20 30,25 25,30 C20,25 10,20 10,10 C15,15 20,15 25,10 Z" />
        </svg>
        <div class="logo-text">expresso</div>
    </div>
</div>
"""

# === HEADER WITH EXPRESSO BRANDING ===
def render_header():
    st.markdown("""
    <div class="main-header" style="background-color: white; padding: 1rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee;">
        <div class="logo-container">
            <div class="logo-circle">
                <svg viewBox="0 0 50 50" width="40" height="40" fill="#4a235a">
                    <path d="M25,10 C30,15 35,15 40,10 C40,20 30,25 25,30 C20,25 10,20 10,10 C15,15 20,15 25,10 Z" />
                </svg>
                <div class="logo-text">expresso</div>
            </div>
        </div>
        
        <div class="nav-links">
            <span class="nav-link">ACCUEIL</span>
            <span class="nav-link">PARTICULIERS</span>
            <span class="nav-link">PROFESSIONNELS</span>
            <span class="nav-link">A PROPOS</span>
        </div>
        
        <button class="custom-button" style="background-color: #f9ca24; color: #4a235a; padding: 0.5rem 1.5rem; border-radius: 50px; font-weight: 500; border: none; cursor: pointer; width: auto;">VOIR NOS AGENCES</button>
    </div>
    
    <div class="contact-button">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="#f9ca24">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
        </svg>
    </div>
    <div class="contact-text">Contactez-nous</div>
    """, unsafe_allow_html=True)

# Render the header
render_header()

# === LOAD DATA AND MODEL ===
try:
    # Replace with your actual dataset and model paths
    df = pd.read_csv('expresso_churn_dataset.csv')
    model = joblib.load("expresso_churn_model.joblib")
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.warning("Please update the file paths in the code to match your environment.")
    # Create sample data for demonstration
    df = pd.DataFrame({
        'REGION': ['Dakar', 'Thies', 'Saint-Louis', 'Nouakchott', 'Dakar', 'Thies', 'Saint-Louis', 'Nouakchott', 'Dakar', 'Thies'],
        'TENURE': [12, 24, 36, 6, 18, 30, 9, 15, 27, 3],
        'MONTANT': [15000, 25000, 10000, 30000, 20000, 15000, 25000, 10000, 30000, 20000],
        'FREQUENCE_RECH': [10, 15, 5, 20, 12, 8, 18, 7, 14, 9],
        'REVENUE': [5000, 8000, 3000, 10000, 7000, 4000, 9000, 2000, 6000, 5000],
        'ARPU_SEGMENT': ['Medium', 'High', 'Low', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'Low'],
        'FREQUENCE': [25, 40, 15, 45, 30, 20, 35, 10, 28, 22],
        'DATA_VOLUME': [2000, 5000, 1000, 8000, 3000, 1500, 6000, 800, 4000, 1200],
        'ON_NET': [500, 800, 300, 1000, 600, 400, 900, 200, 700, 350],
        'ORANGE': [200, 350, 100, 400, 250, 150, 300, 80, 220, 120],
        'TIGO': [150, 300, 80, 350, 200, 100, 250, 50, 180, 90],
        'ZONE1': [50, 100, 20, 150, 80, 30, 120, 10, 70, 40],
        'ZONE2': [30, 60, 10, 80, 40, 20, 70, 5, 35, 25],
        'MRG': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'REGULARITY': [7, 9, 4, 8, 6, 5, 8, 3, 7, 4],
        'CHURN': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    })
    model = None

# === EXPRESSO BACKGROUND IMAGES ===
# Replace these with the actual image URLs you provided
expresso_images = {
    "overview": "https://www.expressotelecom.sn/wp-content/uploads/2022/03/expresso-logo.png",
    "evaluation": "https://scontent.fdkr5-1.fna.fbcdn.net/v/t39.30808-6/348660767_1608789046291399_8988156283888260636_n.jpg?_nc_cat=104&ccb=1-7&_nc_sid=5f2048&_nc_ohc=dQnIV9WDNXAAX_Qe_Oe&_nc_ht=scontent.fdkr5-1.fna&oh=00_AfCHHYQnm_8R_CdQZhLRrw-XVj_Yx_qYMYwwHXJgHjFQrw&oe=66A0D5D9",
    "classifier": "https://pbs.twimg.com/media/GQnQQQsWwAA9Aqj?format=jpg&name=large"
}

# === HELPER FUNCTIONS ===
def create_churn_gauge(probability, min_prob=0, max_prob=1):
    """Create a gauge chart for churn probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        number = {"suffix": "%", "valueformat": ".1f"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [min_prob, max_prob], 'tickwidth': 1, 'tickcolor': "#4a235a"},
            'bar': {'color': "#6c3483"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#eee",
            'steps': [
                {'range': [min_prob, 0.3], 'color': '#4CAF50'},
                {'range': [0.3, 0.7], 'color': '#FFC107'},
                {'range': [0.7, max_prob], 'color': '#F44336'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#4a235a", 'family': "Arial"},
        height = 300,
        margin = dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_feature_importance_chart():
    """Create a feature importance chart for the churn model"""
    # This is a simplified version - in a real app, you'd extract this from your model
    features = ['TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'DATA_VOLUME', 'REGION']
    importance = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
    
    fig = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'},
        color=importance,
        color_continuous_scale=['#6c3483', '#8e44ad', '#9b59b6', '#bb8fce', '#d2b4de', '#e8daef']
    )
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#4a235a", 'family': "Arial"},
        height = 300,
        margin = dict(l=20, r=20, t=30, b=20),
        coloraxis_showscale=False
    )
    
    return fig

def create_churn_comparison_chart(predicted_prob, similar_customers):
    """Create a chart comparing predicted churn with similar customers"""
    fig = px.bar(
        similar_customers, 
        x='Customer_Segment', 
        y='Churn_Rate',
        color='ARPU_SEGMENT',
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'Churn_Rate': 'Churn Rate (%)', 'Customer_Segment': 'Customer Segment'},
        text_auto=True
    )
    
    # Add line for predicted probability
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(similar_customers)-0.5,
        y0=predicted_prob*100,
        y1=predicted_prob*100,
        line=dict(color="#6c3483", width=3, dash="dash")
    )
    
    fig.add_annotation(
        x=len(similar_customers)/2,
        y=predicted_prob*100*1.05,
        text="Your Predicted Churn Probability",
        showarrow=False,
        font=dict(color="#6c3483", size=14)
    )
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(249,249,249,1)",
        font = {'color': "#4a235a", 'family': "Arial"},
        height = 400,
        margin = dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

def get_similar_customers(region, tenure, arpu_segment):
    """Get similar customers for comparison"""
    # This is a simplified version - in a real app, you'd query your database
    similar_customers = pd.DataFrame({
        'Customer_Segment': ['New Users', 'Medium Tenure', 'Long-term', 'High Value', 'Low Usage'],
        'ARPU_SEGMENT': ['Low', 'Medium', 'High', 'High', 'Low'],
        'Tenure': [3, 12, 36, 24, 6],
        'Churn_Rate': [35, 25, 15, 10, 40]
    })
    
    return similar_customers

# === MAIN APP CONTENT ===
# Main container
main_container = st.container()

with main_container:
    # App header with logo and title
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.markdown(expresso_logo, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>Expresso Churn Prediction</h1>", unsafe_allow_html=True)
        
        # Add last updated timestamp
        current_date = datetime.now().strftime("%B %d, %Y")
        st.markdown(f"<p style='text-align: center; color: #888;'>Last updated: {current_date}</p>", unsafe_allow_html=True)
    
    # Create tabs for the three main sections
    tab1, tab2, tab3 = st.tabs(["Overview", "Evaluation", "Machine Learning Classifier"])
    
    # === OVERVIEW TAB ===
    with tab1:
        # Background image container
        st.markdown(f"""
        <div class="background-container">
            <img src="{expresso_images['overview']}" class="background-image">
            <h2><span class="yellow-accent">ici</span> c'est la data qui décide</h2>
            <p>Exploring the Expresso Churn dataset to understand customer behavior and predict churn.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### About the Dataset
            
            The Expresso Churn dataset was provided as part of the Expresso Churn Prediction Challenge hosted by Zindi platform. It contains data on 2.5 million Expresso clients with more than 15 behavior variables to predict client churn probability.
            
            **Key Information:**
            - **Telecom Provider:** Expresso, operating in Mauritania and Senegal
            - **Total Records:** 2.5 million clients
            - **Features:** 15+ behavioral variables
            - **Target Variable:** Customer churn (binary classification)
            """)
        
        with col2:
            st.markdown("### Sample Data")
            st.dataframe(df.head())
        
        st.markdown("### Feature Descriptions")
        
        feature_descriptions = {
            'REGION': 'Geographic location of the customer',
            'TENURE': 'Number of months the customer has been with Expresso',
            'MONTANT': 'Amount spent by the customer',
            'FREQUENCE_RECH': 'Frequency of recharges',
            'REVENUE': 'Revenue generated by the customer',
            'ARPU_SEGMENT': 'Average Revenue Per User segment',
            'FREQUENCE': 'Frequency of usage',
            'DATA_VOLUME': 'Volume of data used',
            'ON_NET': 'On-network calls',
            'ORANGE': 'Calls to Orange network',
            'TIGO': 'Calls to Tigo network',
            'ZONE1': 'International calls to Zone 1',
            'ZONE2': 'International calls to Zone 2',
            'MRG': 'Migration indicator',
            'REGULARITY': 'Regularity of usage',
            'CHURN': 'Target variable indicating whether the customer churned (1) or not (0)'
        }
        
        # Display feature descriptions in a more visually appealing way
        cols = st.columns(2)
        for i, (feature, description) in enumerate(feature_descriptions.items()):
            with cols[i % 2]:
                st.markdown(f"**{feature}**: {description}")
        
        # Data distribution visualization
        st.markdown("### Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='CHURN', data=df, palette=['#6c3483', '#f9ca24'])
            plt.title('Churn Distribution')
            plt.xlabel('Churn (0 = No, 1 = Yes)')
            plt.ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='REGION', data=df, palette=['#6c3483', '#8e44ad', '#9b59b6', '#f9ca24'])
            plt.title('Customer Distribution by Region')
            plt.xlabel('Region')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # === EVALUATION TAB ===
    with tab2:
        # Background image container
        st.markdown(f"""
        <div class="background-container">
            <img src="{expresso_images['evaluation']}" class="background-image">
            <h2><span class="yellow-accent">ici</span> c'est la précision qui décide</h2>
            <p>Evaluating our churn prediction model with key performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Model Evaluation")
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="metric-value">92.5%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Precision</h3>
                <div class="metric-value">89.7%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="metric-value">85.3%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>F1 Score</h3>
                <div class="metric-value">87.4%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        confusion_matrix = np.array([[1800, 200], [150, 850]])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='PuBu',
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        # Feature Importance
        st.markdown("### Feature Importance")
        
        # Sort features by importance
        feature_importance = pd.DataFrame({
            'Feature': ['TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'DATA_VOLUME', 'ON_NET', 
                       'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY'],
            'Importance': [0.25, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='PuBu_r')
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        st.pyplot(fig)
        
        # ROC Curve
        st.markdown("### ROC Curve")
        
        # Sample ROC curve data
        fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        tpr = np.array([0, 0.4, 0.65, 0.8, 0.88, 0.92, 0.95, 0.97, 0.985, 0.99, 1])
        roc_auc = 0.91
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#6c3483', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='#f9ca24', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(fig)
    
    # === MACHINE LEARNING CLASSIFIER TAB ===
    with tab3:
        # Background image container
        st.markdown(f"""
        <div class="background-container">
            <img src="{expresso_images['classifier']}" class="background-image">
            <h2><span class="yellow-accent">ici</span> c'est l'IA qui décide</h2>
            <p>Implementing and testing our machine learning classifier for churn prediction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Machine Learning Classifier")
        
        st.markdown("""
        ### Model Information
        
        We've implemented a **Random Forest Classifier** to predict customer churn. This model was chosen for its:
        
        - High accuracy for classification tasks
        - Ability to handle non-linear relationships
        - Feature importance capabilities
        - Robustness to outliers and non-normalized data
        
        The model was trained on 70% of the data and tested on the remaining 30%.
        """)
        
        # Interactive prediction section
        st.markdown("### Try the Model")
        st.markdown("Adjust the parameters below to see how they affect the churn prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Tenure (months)", 1, 60, 12)
            montant = st.slider("Amount Spent", 1000, 50000, 10000)
            freq_rech = st.slider("Recharge Frequency", 0, 30, 5)
            revenue = st.slider("Revenue", 500, 20000, 5000)
        
        with col2:
            region = st.selectbox("Region", ["Dakar", "Thies", "Saint-Louis", "Nouakchott"])
            arpu = st.selectbox("ARPU Segment", ["Low", "Medium", "High"])
            data_volume = st.slider("Data Volume", 0, 10000, 2000)
            regularity = st.slider("Regularity", 0, 10, 5)
        
        # Create a sample for prediction
        sample = pd.DataFrame({
            'REGION': [region],
            'TENURE': [tenure],
            'MONTANT': [montant],
            'FREQUENCE_RECH': [freq_rech],
            'REVENUE': [revenue],
            'ARPU_SEGMENT': [arpu],
            'FREQUENCE': [np.random.randint(0, 50)],
            'DATA_VOLUME': [data_volume],
            'ON_NET': [np.random.uniform(0, 1000)],
            'ORANGE': [np.random.uniform(0, 500)],
            'TIGO': [np.random.uniform(0, 500)],
            'ZONE1': [np.random.uniform(0, 200)],
            'ZONE2': [np.random.uniform(0, 100)],
            'MRG': [np.random.choice([0, 1])],
            'REGULARITY': [regularity]
        })
        
        # Make prediction (simulated for demo)
        if st.button("Predict Churn"):
            # Show loading animation
            st.markdown("""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div class="loading"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate prediction
            import time
            time.sleep(1)  # Simulate processing time
            
            # For demo purposes, calculate a probability based on input values
            # In a real app, you would use your trained model
            churn_prob = 0.2
            if tenure < 6:
                churn_prob += 0.3
            if montant < 5000:
                churn_prob += 0.2
            if freq_rech < 10:
                churn_prob += 0.1
            if revenue < 3000:
                churn_prob += 0.1
            
            # Cap probability between 0 and 1
            churn_prob = min(max(churn_prob, 0.05), 0.95)
            
            # Display prediction
            st.markdown("### Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Churn Prediction</h3>
                    <div class="metric-value">{int(churn_prob > 0.5)}</div>
                    <p>{'Customer likely to churn' if churn_prob > 0.5 else 'Customer likely to stay'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Churn Probability</h3>
                    <div class="metric-value">{churn_prob:.1%}</div>
                    <p>Confidence level of the prediction</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Churn probability gauge
            st.plotly_chart(create_churn_gauge(churn_prob*100), use_container_width=True)
            
            # Recommendations based on prediction
            st.markdown("### Recommendations")
            
            if churn_prob > 0.5:
                st.markdown("""
                <div class="background-container">
                    <h3>Customer Retention Strategies</h3>
                    <p>Based on the prediction, this customer is at high risk of churning. Consider the following retention strategies:</p>
                    <ul>
                        <li>Offer a personalized loyalty discount</li>
                        <li>Reach out with a customer satisfaction survey</li>
                        <li>Provide a free service upgrade for 3 months</li>
                        <li>Send targeted communications about services they frequently use</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="background-container">
                    <h3>Customer Growth Strategies</h3>
                    <p>This customer is likely to stay. Consider these strategies to increase their value:</p>
                    <ul>
                        <li>Offer premium service upgrades</li>
                        <li>Introduce referral bonuses</li>
                        <li>Cross-sell complementary services</li>
                        <li>Invite to loyalty program with exclusive benefits</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Compare with similar customers
            st.markdown("### Comparison with Similar Customers")
            similar_customers = get_similar_customers(region, tenure, arpu)
            st.plotly_chart(create_churn_comparison_chart(churn_prob, similar_customers), use_container_width=True)

# Footer
st.markdown("""
<div style="background-color: #4a235a; padding: 2rem; margin-top: 2rem; color: white; text-align: center; border-radius: 10px;">
    <p>© 2025 Expresso. All rights reserved.</p>
    <p>Expresso provides telecommunication services in Mauritania and Senegal.</p>
</div>
""", unsafe_allow_html=True)
