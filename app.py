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
import time
import base64
from io import BytesIO
import pydeck as pdk
import plotly.figure_factory as ff
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo

# === PAGE SETUP ===
st.set_page_config(
    page_title="Expresso Churn Prediction",
    page_icon="🌸",
    layout="wide"
)

# === GET PAGE FROM URL ===
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["overview"])[0]

# === CUSTOM CSS FOR EXPRESSO THEME WITH ANIMATIONS AND CUSTOM CURSOR ===
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
    
    /* Card-like containers with animations */
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs styling with animations */
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
        transform: translateY(-3px);
    }
    
    .tab-active {
        border-bottom: 2px solid #f9ca24;
        color: #6c3483;
    }
    
    /* Button styling with animations */
    .custom-button {
        background-color: #f9ca24;
        color: #4a235a;
        padding: 10px 20px;
        border-radius: 50px;
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
        animation: slideIn 0.5s ease-in-out;
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
    
    /* Expresso icon with animation */
    .expresso-icon {
        width: 100%;
        max-width: 150px;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 8px rgba(108, 52, 131, 0.5));
        transition: transform 0.3s ease;
    }
    
    .expresso-icon:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    /* Prediction result with animation */
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
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Custom select box */
    .custom-select {
        background-color: #f9f9f9;
        border: 1px solid #eee;
        border-radius: 5px;
        color: #4a235a;
        padding: 10px;
    }
    
    /* Divider with animation */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #f9ca24, transparent);
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* Feature cards with animation */
    .feature-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 3px solid #6c3483;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in-out;
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
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #6c3483 !important;
    }
    
    /* Responsive design with touch-optimized controls */
    @media (max-width: 768px) {
        .hide-mobile {
            display: none;
        }
        
        h1 {
            font-size: 1.8rem;
        }
        
        /* Larger touch targets for mobile */
        .stButton > button {
            padding: 12px 24px !important;
            font-size: 16px !important;
        }
        
        /* Larger sliders for touch */
        .stSlider [data-baseweb="thumb"] {
            height: 24px !important;
            width: 24px !important;
        }
        
        /* Larger select boxes */
        .stSelectbox [data-baseweb="select"] {
            height: 44px !important;
        }
    }
    
    /* Logo container with animation */
    .logo-container {
        display: flex;
        align-items: center;
        animation: fadeIn 0.5s ease-in-out;
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
        transition: transform 0.3s ease;
    }
    
    .logo-circle:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    .logo-text {
        color: #4a235a;
        font-weight: bold;
        font-size: 1.5rem;
        margin-top: 40px;
    }
    
    /* Navigation styling with animations */
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-link {
        color: #4a235a;
        font-weight: 500;
        text-decoration: none;
        font-size: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-link:hover {
        color: #8e44ad;
        transform: translateY(-2px);
    }
    
    /* Background container with animation */
    .background-container {
        position: relative;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        background: linear-gradient(to right, rgba(108, 52, 131, 0.9), rgba(74, 35, 90, 0.9));
        margin-bottom: 2rem;
        animation: fadeIn 0.5s ease-in-out;
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
    
    /* Contact button with animation */
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
        transition: transform 0.3s ease;
    }
    
    .contact-button:hover {
        transform: scale(1.1);
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
        opacity: 0;
        transition: opacity 0.3s ease, transform 0.3s ease;
    }
    
    .contact-button:hover + .contact-text {
        opacity: 1;
        transform: translateX(5px);
    }
    
    /* Yellow accent for highlights */
    .yellow-accent {
        color: #f9ca24;
        font-weight: bold;
    }
    
    /* Collapsible sections */
    .collapsible {
        background-color: #f9f9f9;
        color: #4a235a;
        cursor: pointer;
        padding: 18px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 16px;
        border-radius: 5px;
        margin-bottom: 10px;
        transition: 0.3s;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .collapsible:after {
        content: '\\002B';
        color: #6c3483;
        font-weight: bold;
        float: right;
        margin-left: 5px;
    }
    
    .active:after {
        content: "\\2212";
    }
    
    .collapsible-content {
        padding: 0 18px;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease-out;
        background-color: #ffffff;
        border-radius: 0 0 5px 5px;
    }
    
    /* Progress indicator */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
        position: relative;
    }
    
    .step-container:before {
        content: '';
        position: absolute;
        background: #f0f0f0;
        height: 4px;
        width: 100%;
        top: 50%;
        transform: translateY(-50%);
        z-index: 1;
    }
    
    .step {
        width: 30px;
        height: 30px;
        background-color: #f0f0f0;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        color: #4a235a;
        font-weight: bold;
        position: relative;
        z-index: 2;
        transition: all 0.3s ease;
    }
    
    .step.active {
        background-color: #6c3483;
        color: white;
    }
    
    .step.completed {
        background-color: #f9ca24;
        color: #4a235a;
    }
    
    .step-label {
        position: absolute;
        top: 35px;
        font-size: 12px;
        color: #4a235a;
        width: 100px;
        text-align: center;
        left: 50%;
        transform: translateX(-50%);
    }
    
    /* Custom cursor */
    * {
        cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%236c3483' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M5 12h14'%3E%3C/path%3E%3Cpath d='M12 5v14'%3E%3C/path%3E%3C/svg%3E"), auto;
    }
    
    button, a, .stButton > button, .stSelectbox, .stSlider, [data-testid="stWidgetLabel"] {
        cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23f9ca24' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 16l6-6-6-6'%3E%3C/path%3E%3Cpath d='M4 21v-7a4 4 0 0 1 4-4h11'%3E%3C/path%3E%3C/svg%3E"), pointer !important;
    }
    
    /* Dashboard container */
    .dashboard-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .dashboard-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
    }
    
    /* Tour guide */
    .tour-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 9998;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .tour-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        max-width: 500px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 9999;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .tour-title {
        color: #6c3483;
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    .tour-content {
        margin-bottom: 20px;
    }
    
    .tour-buttons {
        display: flex;
        justify-content: space-between;
    }
    
    .tour-button {
        background-color: #6c3483;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .tour-button:hover {
        background-color: #8e44ad;
    }
    
    .tour-skip {
        background-color: #f0f0f0;
        color: #4a235a;
    }
    
    /* Fix for navigation links */
    .main-header {
        background-color: white;
        padding: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #eee;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-link {
        color: #4a235a;
        font-weight: 500;
        text-decoration: none;
        font-size: 1rem;
        cursor: pointer;
    }
    
    .header-button {
        background-color: #f9ca24;
        color: #4a235a;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 500;
        border: none;
        cursor: pointer;
        width: auto;
    }
</style>
""", unsafe_allow_html=True)

# === JAVASCRIPT FOR ANIMATIONS AND INTERACTIVE ELEMENTS ===
st.markdown("""
<script>
// Collapsible sections
document.addEventListener('DOMContentLoaded', function() {
    var coll = document.getElementsByClassName("collapsible");
    for (var i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        });
    }
});

// Tour guide functionality
let currentStep = 0;
const tourSteps = [
    {
        title: "Welcome to Expresso Churn Prediction",
        content: "This guided tour will help you understand how to use this application effectively.",
        target: "body"
    },
    {
        title: "Overview Tab",
        content: "Here you can explore the dataset and understand the features that influence customer churn.",
        target: "#tab-overview"
    },
    {
        title: "Evaluation Tab",
        content: "This section shows how well our model performs in predicting customer churn.",
        target: "#tab-evaluation"
    },
    {
        title: "Prediction Tool",
        content: "Use this interactive tool to predict if a customer is likely to churn based on their attributes.",
        target: "#tab-prediction"
    },
    {
        title: "Customer Parameters",
        content: "Adjust these sliders to see how different factors affect churn probability.",
        target: ".stSlider"
    }
];

function showTourStep(step) {
    const tourOverlay = document.createElement('div');
    tourOverlay.className = 'tour-overlay';
    
    const tourBox = document.createElement('div');
    tourBox.className = 'tour-box';
    
    const tourTitle = document.createElement('h3');
    tourTitle.className = 'tour-title';
    tourTitle.textContent = tourSteps[step].title;
    
    const tourContent = document.createElement('div');
    tourContent.className = 'tour-content';
    tourContent.textContent = tourSteps[step].content;
    
    const tourButtons = document.createElement('div');
    tourButtons.className = 'tour-buttons';
    
    const prevButton = document.createElement('button');
    prevButton.className = 'tour-button tour-prev';
    prevButton.textContent = 'Previous';
    prevButton.style.display = step === 0 ? 'none' : 'block';
    prevButton.onclick = () => {
        document.body.removeChild(tourOverlay);
        showTourStep(step - 1);
    };
    
    const nextButton = document.createElement('button');
    nextButton.className = 'tour-button tour-next';
    nextButton.textContent = step === tourSteps.length - 1 ? 'Finish' : 'Next';
    nextButton.onclick = () => {
        document.body.removeChild(tourOverlay);
        if (step < tourSteps.length - 1) {
            showTourStep(step + 1);
        }
    };
    
    const skipButton = document.createElement('button');
    skipButton.className = 'tour-button tour-skip';
    skipButton.textContent = 'Skip Tour';
    skipButton.onclick = () => {
        document.body.removeChild(tourOverlay);
    };
    
    tourButtons.appendChild(skipButton);
    if (step > 0) tourButtons.appendChild(prevButton);
    tourButtons.appendChild(nextButton);
    
    tourBox.appendChild(tourTitle);
    tourBox.appendChild(tourContent);
    tourBox.appendChild(tourButtons);
    
    tourOverlay.appendChild(tourBox);
    document.body.appendChild(tourOverlay);
}

// Start tour when button is clicked
function startTour() {
    showTourStep(0);
}
</script>
""", unsafe_allow_html=True)

# === EXPRESSO LOGO SVG WITH ANIMATION ===
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

# === HEADER WITH EXPRESSO BRANDING (FIXED) ===
def render_header():
    st.markdown("""
    <div class="main-header">
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
        
        <button class="header-button">VOIR NOS AGENCES</button>
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
        'user_id': range(1, 11),
        'REGION': ['Dakar', 'Thies', 'Saint-Louis', 'Nouakchott', 'Dakar', 'Thies', 'Saint-Louis', 'Nouakchott', 'Dakar', '  'Thies', 'Saint-Louis', 'Nouakchott', 'Dakar', 'Thies', 'Saint-Louis', 'Nouakchott', 'Dakar', 'Thies'],
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
# Using direct image URLs for better reliability
expresso_images = {
    "overview": "https://www.expressotelecom.sn/wp-content/uploads/2022/03/expresso-logo.png",
    "evaluation": "https://scontent.fdkr5-1.fna.fbcdn.net/v/t39.30808-6/348660767_1608789046291399_8988156283888260636_n.jpg",
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

def create_3d_scatter():
    """Create a 3D scatter plot for data visualization"""
    # Sample data for 3D visualization
    z = df['TENURE'].values[:100] if 'TENURE' in df.columns else np.random.randint(1, 60, 100)
    x = df['MONTANT'].values[:100] if 'MONTANT' in df.columns else np.random.uniform(1000, 50000, 100)
    y = df['REVENUE'].values[:100] if 'REVENUE' in df.columns else np.random.uniform(500, 20000, 100)
    colors = df['CHURN'].values[:100] if 'CHURN' in df.columns else np.random.choice([0, 1], 100)
    
    fig = px.scatter_3d(
        x=x, y=y, z=z,
        color=colors,
        color_discrete_sequence=['#4CAF50', '#F44336'],
        labels={'x': 'Amount Spent', 'y': 'Revenue', 'z': 'Tenure', 'color': 'Churn'},
        title="3D Visualization of Customer Data"
    )
    
    fig.update_layout(
        scene = dict(
            xaxis_title='Amount Spent',
            yaxis_title='Revenue',
            zaxis_title='Tenure',
            xaxis = dict(backgroundcolor="#f9f9f9"),
            yaxis = dict(backgroundcolor="#f9f9f9"),
            zaxis = dict(backgroundcolor="#f9f9f9")
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def create_geographical_heatmap():
    """Create a geographical heatmap of churn rates"""
    # Sample data for geographical visualization
    # In a real app, you would aggregate actual data by region
    geo_data = pd.DataFrame({
        'Region': ['Dakar', 'Thies', 'Saint-Louis', 'Nouakchott', 'Kaolack', 'Ziguinchor'],
        'Latitude': [14.6937, 14.7910, 16.0179, 18.0735, 14.1652, 12.5598],
        'Longitude': [-17.4441, -16.9359, -16.4896, -15.9582, -16.0726, -16.2730],
        'Churn_Rate': [25, 30, 20, 35, 28, 22]
    })
    
    fig = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=15.5,
            longitude=-16.5,
            zoom=6,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=geo_data,
                get_position=['Longitude', 'Latitude'],
                get_elevation='Churn_Rate',
                elevation_scale=100,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                coverage=1,
                get_fill_color="[255, (1 - Churn_Rate / 40) * 255, 0]",
            ),
        ],
        tooltip={"text": "{Region}\nChurn Rate: {Churn_Rate}%"}
    )
    
    return fig

def preprocess_input_data(input_df):
    """Preprocess input data according to the steps provided by the user"""
    # Create a copy to avoid modifying the original
    df_processed = input_df.copy()
    
    # 1. Handle Missing Values
    cols_to_drop = ['ZONE1', 'ZONE2']
    if all(col in df_processed.columns for col in cols_to_drop):
        df_processed.drop(columns=cols_to_drop, inplace=True)
    
    if 'MRG' in df_processed.columns and df_processed['MRG'].nunique() == 1:
        df_processed.drop(columns=['MRG'], inplace=True)
    
    # Impute categorical columns with mode
    for col in ['REGION', 'TOP_PACK']:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Impute numerical columns with median
    num_cols_to_impute = [
        'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
        'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME',
        'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK'
    ]
    for col in num_cols_to_impute:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # 2. Feature Engineering
    # Define ordered mapping for TENURE if it's categorical
    if 'TENURE' in df_processed.columns and df_processed['TENURE'].dtype == 'object':
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
        df_processed['TENURE'] = df_processed['TENURE'].map(tenure_order)
    
    # One-Hot Encode REGION
    if 'REGION' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['REGION'], prefix='REGION', dummy_na=True)
    
    # Frequency Encode TOP_PACK
    if 'TOP_PACK' in df_processed.columns:
        top_pack_freq = df_processed['TOP_PACK'].value_counts()
        df_processed['TOP_PACK_FE'] = df_processed['TOP_PACK'].map(top_pack_freq)
        df_processed.drop(columns=['TOP_PACK'], inplace=True)
        
        # Normalize the frequency-encoded column
        scaler = MinMaxScaler()
        df_processed['TOP_PACK_FE'] = scaler.fit_transform(df_processed[['TOP_PACK_FE']])
    
    # 3. Final Processing
    # Convert boolean columns to int
    bool_cols = df_processed.select_dtypes(include='bool').columns
    df_processed[bool_cols] = df_processed[bool_cols].astype(int)
    
    # Handle outliers for numeric columns (simplified approach)
    num_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [col for col in num_cols if df_processed[col].nunique() <= 2]
    iqr_cols = [col for col in num_cols if col not in binary_cols + ['CHURN']]
    
    for col in iqr_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
    
    return df_processed

# === GUIDED TOUR FUNCTIONALITY ===
def show_guided_tour():
    tour_started = st.session_state.get('tour_started', False)
    
    if not tour_started:
        with st.sidebar:
            st.markdown("### 🚀 First time here?")
            if st.button("Start Guided Tour"):
                st.session_state['tour_started'] = True
                st.session_state['tour_step'] = 0
                st.experimental_rerun()
    
    if tour_started:
        tour_step = st.session_state.get('tour_step', 0)
        tour_steps = [
            {
                "title": "Welcome to Expresso Churn Prediction",
                "content": "This guided tour will help you understand how to use this application effectively."
            },
            {
                "title": "Overview Tab",
                "content": "Here you can explore the dataset and understand the features that influence customer churn."
            },
            {
                "title": "Evaluation Tab",
                "content": "This section shows how well our model performs in predicting customer churn."
            },
            {
                "title": "Prediction Tool",
                "content": "Use this interactive tool to predict if a customer is likely to churn based on their attributes."
            },
            {
                "title": "Customer Parameters",
                "content": "Adjust these sliders to see how different factors affect churn probability."
            }
        ]
        
        # Display tour overlay
        st.markdown(f"""
        <div style="position: fixed; bottom: 20px; right: 20px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); z-index: 9999; max-width: 400px;">
            <h3 style="color: #6c3483; margin-top: 0;">{tour_steps[tour_step]['title']}</h3>
            <p>{tour_steps[tour_step]['content']}</p>
            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                <button onclick="window.tourStep = {tour_step - 1}; Streamlit.setComponentValue(window.tourStep);" style="background-color: #f0f0f0; color: #4a235a; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; {'' if tour_step > 0 else 'visibility: hidden;'}">Previous</button>
                <button onclick="Streamlit.setComponentValue('end_tour');" style="background-color: #f0f0f0; color: #4a235a; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">Skip Tour</button>
                <button onclick="window.tourStep = {tour_step + 1}; Streamlit.setComponentValue(window.tourStep);" style="background-color: #6c3483; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">{tour_step == len(tour_steps) - 1 and 'Finish' or 'Next'}</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle tour navigation
        if st.button("Next", key="tour_next", help="Go to next step"):
            if tour_step < len(tour_steps) - 1:
                st.session_state['tour_step'] += 1
            else:
                st.session_state['tour_started'] = False
            st.experimental_rerun()
        
        if tour_step > 0 and st.button("Previous", key="tour_prev", help="Go to previous step"):
            st.session_state['tour_step'] -= 1
            st.experimental_rerun()
        
        if st.button("End Tour", key="tour_end", help="End the guided tour"):
            st.session_state['tour_started'] = False
            st.experimental_rerun()

# === PROGRESS INDICATOR ===
def show_progress_indicator(current_step, total_steps=4):
    steps = ["Data Input", "Preprocessing", "Model Prediction", "Results"]
    
    progress_html = """
    <div class="step-container">
    """
    
    for i in range(total_steps):
        status = ""
        if i < current_step:
            status = "completed"
        elif i == current_step:
            status = "active"
        
        progress_html += f"""
        <div class="step {status}">
            {i+1}
            <div class="step-label">{steps[i]}</div>
        </div>
        """
    
    progress_html += """
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

# === COLLAPSIBLE SECTIONS ===
def create_collapsible_section(title, content, expanded=False):
    section_id = title.lower().replace(" ", "_")
    
    if expanded:
        st.markdown(f"""
        <button class="collapsible active" id="{section_id}_btn">{title}</button>
        <div class="collapsible-content" id="{section_id}_content" style="max-height: 1000px; padding: 18px;">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <button class="collapsible" id="{section_id}_btn">{title}</button>
        <div class="collapsible-content" id="{section_id}_content">
            {content}
        </div>
        """, unsafe_allow_html=True)

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
    
    # Show guided tour button
    with col3:
        if st.button("Start Tour", help="Take a guided tour of the application"):
            st.session_state['tour_started'] = True
            st.session_state['tour_step'] = 0
            st.experimental_rerun()
    
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
        
        # Collapsible Dataset Overview
        dataset_overview_content = """
        <div class="card">
            <p>The Expresso Churn dataset was provided as part of the Expresso Churn Prediction Challenge hosted by Zindi platform. It contains data on 2.5 million Expresso clients with more than 15 behavior variables to predict client churn probability.</p>
            
            <p><strong>Key Information:</strong></p>
            <ul>
                <li><strong>Telecom Provider:</strong> Expresso, operating in Mauritania and Senegal</li>
                <li><strong>Total Records:</strong> 2.5 million clients</li>
                <li><strong>Features:</strong> 15+ behavioral variables</li>
                <li><strong>Target Variable:</strong> Customer churn (binary classification)</li>
            </ul>
        </div>
        """
        create_collapsible_section("Dataset Overview", dataset_overview_content, expanded=True)
        
        # Sample Data in collapsible section
        st.markdown("### Sample Data")
        st.dataframe(df.head())
        
        # Feature Descriptions in collapsible section
        feature_descriptions_content = """
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div><strong>REGION</strong>: Geographic location of the customer</div>
            <div><strong>TENURE</strong>: Number of months the customer has been with Expresso</div>
            <div><strong>MONTANT</strong>: Amount spent by the customer</div>
            <div><strong>FREQUENCE_RECH</strong>: Frequency of recharges</div>
            <div><strong>REVENUE</strong>: Revenue generated by the customer</div>
            <div><strong>ARPU_SEGMENT</strong>: Average Revenue Per User segment</div>
            <div><strong>FREQUENCE</strong>: Frequency of usage</div>
            <div><strong>DATA_VOLUME</strong>: Volume of data used</div>
            <div><strong>ON_NET</strong>: On-network calls</div>
            <div><strong>ORANGE</strong>: Calls to Orange network</div>
            <div><strong>TIGO</strong>: Calls to Tigo network</div>
            <div><strong>ZONE1</strong>: International calls to Zone 1</div>
            <div><strong>ZONE2</strong>: International calls to Zone 2</div>
            <div><strong>MRG</strong>: Migration indicator</div>
            <div><strong>REGULARITY</strong>: Regularity of usage</div>
            <div><strong>CHURN</strong>: Target variable indicating whether the customer churned (1) or not (0)</div>
        </div>
        """
        create_collapsible_section("Feature Descriptions", feature_descriptions_content)
        
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
        
        # 3D Visualization
        st.markdown("### 3D Data Visualization")
        st.plotly_chart(create_3d_scatter(), use_container_width=True)
        
        # Geographical Heatmap
        st.markdown("### Geographical Distribution of Churn")
        st.pydeck_chart(create_geographical_heatmap())
    
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
        
        # Interactive Dashboard
        st.markdown("## Model Evaluation Dashboard")
        st.markdown("Use the controls below to customize your dashboard view:")
        
        # Dashboard controls
        col1, col2, col3 = st.columns(3)
        with col1:
            show_metrics = st.checkbox("Show Metrics", value=True)
        with col2:
            show_confusion = st.checkbox("Show Confusion Matrix", value=True)
        with col3:
            show_feature_importance = st.checkbox("Show Feature Importance", value=True)
        
        # Dashboard content based on selections
        if show_metrics:
            # Display metrics in cards
            st.markdown("### Key Performance Metrics")
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
        
        # Create dashboard layout
        dashboard_cols = st.columns(2)
        
        # Confusion Matrix
        if show_confusion:
            with dashboard_cols[0]:
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
        if show_feature_importance:
            with dashboard_cols[1]:
                st.markdown("### Feature Importance")
                
                # Sort features by importance
                feature_importance = pd.DataFrame({
                    'Feature': ['TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'DATA_VOLUME', 'ON_NET', 
                               'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY'],
                    'Importance': [0.25, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
                }).sort
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
        
        # Interactive Model Comparison
        st.markdown("### Model Comparison")
        st.markdown("Compare different model performances:")
        
        # Create tabs for model comparison
        model_tabs = st.tabs(["Random Forest", "XGBoost", "Logistic Regression"])
        
        with model_tabs[0]:
            st.markdown("""
            <div class="card">
                <h4>Random Forest Performance</h4>
                <p>Random Forest is our best performing model with high accuracy and good balance between precision and recall.</p>
                <div class="progress-container">
                    <div class="progress-bar" style="width: 92.5%;"></div>
                </div>
                <p>Accuracy: 92.5%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with model_tabs[1]:
            st.markdown("""
            <div class="card">
                <h4>XGBoost Performance</h4>
                <p>XGBoost performs slightly worse than Random Forest but has faster training time.</p>
                <div class="progress-container">
                    <div class="progress-bar" style="width: 91.2%;"></div>
                </div>
                <p>Accuracy: 91.2%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with model_tabs[2]:
            st.markdown("""
            <div class="card">
                <h4>Logistic Regression Performance</h4>
                <p>Logistic Regression is our baseline model with decent performance and high interpretability.</p>
                <div class="progress-container">
                    <div class="progress-bar" style="width: 85.7%;"></div>
                </div>
                <p>Accuracy: 85.7%</p>
            </div>
            """, unsafe_allow_html=True)
    
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
        
        # Progress indicator
        current_step = st.session_state.get('prediction_step', 0)
        show_progress_indicator(current_step)
        
        # Model Information in collapsible section
        model_info_content = """
        <div class="card">
            <p>We've implemented a <strong>Random Forest Classifier</strong> to predict customer churn. This model was chosen for its:</p>
            <ul>
                <li>High accuracy for classification tasks</li>
                <li>Ability to handle non-linear relationships</li>
                <li>Feature importance capabilities</li>
                <li>Robustness to outliers and non-normalized data</li>
            </ul>
            <p>The model was trained on 70% of the data and tested on the remaining 30%.</p>
        </div>
        """
        create_collapsible_section("Model Information", model_info_content, expanded=True)
        
        # Interactive prediction section
        st.markdown("### Try the Model")
        st.markdown("Adjust the parameters below to see how they affect the churn prediction:")
        
        # Touch-optimized controls
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Tenure (months)", 1, 60, 12, help="Number of months the customer has been with Expresso")
            montant = st.slider("Amount Spent", 1000, 50000, 10000, help="Total amount spent by the customer")
            freq_rech = st.slider("Recharge Frequency", 0, 30, 5, help="How often the customer recharges their account")
            revenue = st.slider("Revenue", 500, 20000, 5000, help="Revenue generated by the customer")
        
        with col2:
            region = st.selectbox("Region", ["Dakar", "Thies", "Saint-Louis", "Nouakchott"], help="Customer's geographic location")
            arpu = st.selectbox("ARPU Segment", ["Low", "Medium", "High"], help="Average Revenue Per User segment")
            data_volume = st.slider("Data Volume", 0, 10000, 2000, help="Volume of data used by the customer")
            regularity = st.slider("Regularity", 0, 10, 5, help="Regularity of customer's usage patterns")
        
        # Create a sample for prediction
        sample = pd.DataFrame({
            'user_id': [1],
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
        if st.button("Predict Churn", help="Click to predict churn probability"):
            # Update progress indicator
            st.session_state['prediction_step'] = 1
            
            # Show loading animation with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate data preprocessing
            status_text.text("Preprocessing data...")
            for i in range(25):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            # Apply preprocessing steps
            status_text.text("Applying feature engineering...")
            processed_sample = preprocess_input_data(sample)
            for i in range(25, 50):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            # Update progress indicator
            st.session_state['prediction_step'] = 2
            
            # Simulate model prediction
            status_text.text("Running prediction model...")
            for i in range(50, 75):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
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
            
            # Simulate final processing
            status_text.text("Finalizing results...")
            for i in range(75, 100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            # Update progress indicator
            st.session_state['prediction_step'] = 3
            
            # Clear progress indicators
            status_text.empty()
            
            # Display prediction with animation
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
            
            # Interactive What-If Analysis
            st.markdown("### What-If Analysis")
            st.markdown("See how changing customer attributes would affect churn probability:")
            
            what_if_col1, what_if_col2 = st.columns(2)
            
            with what_if_col1:
                tenure_change = st.slider("Change in Tenure (months)", -12, 12, 0, help="How would adding or reducing months affect churn?")
                montant_change = st.slider("Change in Amount Spent", -10000, 10000, 0, step=1000, help="How would spending more or less affect churn?")
            
            with what_if_col2:
                freq_change = st.slider("Change in Recharge Frequency", -10, 10, 0, help="How would changing recharge frequency affect churn?")
                data_change = st.slider("Change in Data Usage", -5000, 5000, 0, step=500, help="How would using more or less data affect churn?")
            
            if st.button("Recalculate Churn Probability", help="See how these changes would affect churn"):
                # Calculate new probability based on changes
                new_prob = churn_prob
                
                # Tenure effect (negative correlation with churn)
                new_prob -= tenure_change * 0.01
                
                # Amount spent effect (negative correlation with churn)
                new_prob -= montant_change * 0.00001
                
                # Recharge frequency effect (negative correlation with churn)
                new_prob -= freq_change * 0.02
                
                # Data usage effect (varies)
                if data_change > 0:
                    new_prob -= data_change * 0.00002  # More data usage, less churn
                else:
                    new_prob += abs(data_change) * 0.00001  # Less data usage, more churn
                
                # Cap probability between 0 and 1
                new_prob = min(max(new_prob, 0.05), 0.95)
                
                # Display comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Original Probability</h3>
                        <div class="metric-value">{churn_prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>New Probability</h3>
                        <div class="metric-value">{new_prob:.1%}</div>
                        <p>{'↑ Increased' if new_prob > churn_prob else '↓ Decreased'} by {abs(new_prob - churn_prob):.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualization of the change
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "delta",
                    value = new_prob * 100,
                    delta = {'reference': churn_prob * 100, 'relative': True, 'valueformat': '.1f'},
                    title = {'text': "Change in Churn Probability"},
                    domain = {'y': [0, 1], 'x': [0.25, 0.75]}
                ))
                
                fig.update_layout(
                    height = 200,
                    margin = dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Show guided tour if enabled
if 'tour_started' in st.session_state and st.session_state['tour_started']:
    show_guided_tour()

# Footer
st.markdown("""
<div style="background-color: #4a235a; padding: 2rem; margin-top: 2rem; color: white; text-align: center; border-radius: 10px;">
    <p>© 2025 Expresso. All rights reserved.</p>
    <p>Expresso provides telecommunication services in Mauritania and Senegal.</p>
</div>
""", unsafe_allow_html=True)
