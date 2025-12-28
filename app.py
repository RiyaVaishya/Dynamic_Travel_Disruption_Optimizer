# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import datetime
import base64
# Robust SHAP import: catch any exception during import to avoid crashes
SHAP_AVAILABLE = False
SHAP_IMPORT_ERROR = None
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    # Do not raise — record the error and disable SHAP features gracefully
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(e)
    shap = None

# ----------------------------
# Load model + encoders
# ----------------------------
model = joblib.load("compensation_model.pkl")
airline_encoder = joblib.load("airline_encoder.pkl")
ticket_encoder = joblib.load("ticket_encoder.pkl")
region_encoder = joblib.load("region_encoder.pkl")

# ----------------------------
# Page config + modern CSS
# ----------------------------
st.set_page_config(page_title="Dynamic Travel Disruption Optimizer", layout="wide", page_icon="✈️")

# Helper: load local image and return base64 string for embedding in CSS
def _img_to_base64(path: str) -> str:
    try:
        with open(path, 'rb') as _f:
            return base64.b64encode(_f.read()).decode('utf-8')
    except Exception:
        return ''

# encode banner image for CSS background (Streamlit requires data URI for local images)
banner_b64 = _img_to_base64('assets/airport_banner.jpg')

css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            background-attachment: fixed;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        .main-title {
            font-size: 42px;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 12px;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .sub {
            color: #cbd5e1;
            margin-bottom: 32px;
            font-size: 18px;
            font-weight: 400;
            opacity: 0.9;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        
        .kpi-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .kpi-card b {
            color: #e2e8f0;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
        }
        
        .kpi-card div {
            color: #ffffff;
            font-size: 24px;
            font-weight: 600;
            margin-top: 8px;
        }
        
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        section[data-testid="stSidebar"] .css-1d391kg {
            background: transparent;
        }
        
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff;
        }
        
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stNumberInput label,
        section[data-testid="stSidebar"] .stSlider label {
            color: #e2e8f0;
            font-weight: 500;
        }
        
        section[data-testid="stSidebar"] .stSelectbox > div > div,
        section[data-testid="stSidebar"] .stNumberInput > div > div {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        section[data-testid="stSidebar"] .stSlider > div {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }
        
        .stSuccess {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 16px;
            color: #6ee7b7;
        }
        
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 600;
        }
        
        .stMarkdown h3 {
            color: #ffffff;
            margin-top: 32px;
            margin-bottom: 20px;
        }
        
        .stMarkdown hr {
            border-color: rgba(255, 255, 255, 0.1);
            margin: 32px 0;
        }
        
        .stCaption {
            color: #94a3b8;
            opacity: 0.8;
        }
        
        footer {
            visibility: hidden;
        }
        
        .stDataFrame {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
        }
        
        .element-container {
            margin-bottom: 16px;
        }
        
        .hero-banner {
            background-image: linear-gradient(135deg, rgba(5,15,30,0.45) 0%, rgba(2,6,23,0.25) 100%), url('data:image/jpeg;base64,""" + banner_b64 + """');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 56px 32px;
            margin-bottom: 28px;
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.35);
            text-align: center;
        }
        
        .hero-title {
            font-size: 48px;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 16px;
            letter-spacing: -0.5px;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        
        .hero-subtitle {
            color: #e2e8f0;
            font-size: 20px;
            font-weight: 400;
            margin-bottom: 12px;
            opacity: 0.95;
        }
        
        .hero-helper {
            color: #cbd5e1;
            font-size: 16px;
            font-weight: 400;
            opacity: 0.85;
            margin-top: 8px;
        }
        
        .info-strip-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px 20px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .info-strip-card:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        
        .info-strip-icon {
            font-size: 36px;
            margin-bottom: 12px;
            display: block;
        }
        
        .info-strip-title {
            color: #ffffff;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .info-strip-desc {
            color: #cbd5e1;
            font-size: 14px;
            font-weight: 400;
            opacity: 0.9;
            line-height: 1.5;
        }
        
        .stExpander {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            margin: 24px 0;
        }
        
        .stExpander label {
            color: #ffffff;
            font-weight: 600;
        }
        
        .banner-image-container {
            position: relative;
            min-height: 280px;
            background-image: linear-gradient(rgba(5,15,30,0.75), rgba(5,15,30,0.75)), url("assets/airport_banner.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            border-radius: 20px;
            margin: 24px 0;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
        }
        
        .banner-image-content {
            position: relative;
            z-index: 2;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #ffffff;
            padding: 12px;
        }
        
        .banner-image-content .glass {
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            background: rgba(15,23,42,0.6);
            padding: 40px;
            border-radius: 16px;
            max-width: 900px;
            margin: 0 12px;
        }
        
        .banner-image-content h2 {
            margin: 0;
            font-size: 34px;
            line-height: 1.1;
            font-weight: 700;
        }
        
        .banner-image-content p {
            margin-top: 10px;
            font-size: 16px;
            opacity: 0.95;
        }
        
        .feature-importance-container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            margin: 24px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .feature-importance-text {
            color: #cbd5e1;
            font-size: 15px;
            margin-bottom: 20px;
            line-height: 1.6;
        }
    </style>
"""

st.markdown(css, unsafe_allow_html=True)

# ----------------------------
# Main Title (restored exactly)
# ----------------------------
st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>✈️ Dynamic Travel Disruption Compensation Optimizer</div>
        <div class='hero-subtitle'>AI-powered system to estimate fair and transparent compensation for disrupted journeys</div>
        <div class='hero-helper'>Designed for passengers, airlines, and regulators</div>
    </div>
""", unsafe_allow_html=True)

# (removed duplicate introductory card; main title / banner retained)
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Passenger-Centric Info Strip
# ----------------------------
info_col1, info_col2, info_col3 = st.columns([1, 1, 1])
with info_col1:
    st.markdown("""
        <div class='info-strip-card'>
            <span class='info-strip-icon'>⏱</span>
            <div class='info-strip-title'>Delay Impact</div>
            <div class='info-strip-desc'>Understand how flight delays affect compensation calculations</div>
        </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
        <div class='info-strip-card'>
            <span class='info-strip-icon'>💰</span>
            <div class='info-strip-title'>Fair Compensation</div>
            <div class='info-strip-desc'>Get transparent, data-driven estimates based on multiple factors</div>
        </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
        <div class='info-strip-card'>
            <span class='info-strip-icon'>🔍</span>
            <div class='info-strip-title'>Explainable AI</div>
            <div class='info-strip-desc'>See why the model makes specific predictions with clear insights</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Expandable Explainability Section
# ----------------------------
with st.expander("ℹ️ How does this system work?"):
    st.markdown("""
#### 📊 Why Feature Importance Matters

Not all factors contribute equally to the compensation calculation. Feature importance helps us understand which factors (like delay time or ticket class) have the most impact on the final compensation amount. This transparency ensures fairness and helps passengers understand the reasoning behind their compensation.

#### 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) values provide detailed explanations for each prediction. They show exactly how each input feature (delay, distance, loyalty score, etc.) contributes to the final compensation estimate. This makes the AI system transparent, trustworthy, and compliant with regulations that require explainable decision-making.

**Note:** This system is designed to assist in compensation estimation and should be used alongside regulatory guidelines and airline policies.
    """)

# ----------------------------
# Load dataset and basic checks
# ----------------------------
data = pd.read_csv("compensation_data.csv")

expected_cols = ['flight_id','airline','delay_minutes','ticket_class','distance_km','region','loyalty_score','base_compensation','final_compensation']
missing = [c for c in expected_cols if c not in data.columns]
if missing:
    st.error(f"Dataset missing columns: {missing}. Check compensation_data.csv")
    st.stop()

# ----------------------------
# Sidebar - inputs & info
# ----------------------------
st.sidebar.markdown("<div style='padding: 8px 0; margin-bottom: 16px;'><h2 style='color: #ffffff; font-weight: 600; margin: 0;'>🧾 Passenger / Flight Inputs</h2></div>", unsafe_allow_html=True)
st.sidebar.caption("Provide passenger and flight details used for prediction.")
sel_airline = st.sidebar.selectbox("Airline", options=sorted(data['airline'].unique()))
sel_ticket = st.sidebar.selectbox("Ticket Class", options=sorted(data['ticket_class'].unique()))
sel_region = st.sidebar.selectbox("Region", options=sorted(data['region'].unique()))
delay = st.sidebar.number_input("Delay (minutes)", min_value=0, max_value=2000, value=120)
distance = st.sidebar.number_input("Distance (km)", min_value=10, max_value=20000, value=1000)
loyalty = st.sidebar.slider("Loyalty score (0-1)", 0.0, 1.0, 0.5, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='padding: 8px 0;'><div style='color: #cbd5e1; margin: 8px 0;'>Dataset rows: <strong style='color: #ffffff;'>{:,}</strong></div>".format(len(data)), unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #cbd5e1; margin: 8px 0;'>Avg delay: <strong style='color: #ffffff;'>{:.0f}</strong> min</div>".format(data['delay_minutes'].mean()), unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #cbd5e1; margin: 8px 0;'>Avg compensation: <strong style='color: #ffffff;'>₹{:.0f}</strong></div>".format(data['final_compensation'].mean()), unsafe_allow_html=True)

# ----------------------------
# Filter dataset for analytics (dynamic)
# ----------------------------
filtered = data.copy()
if sel_airline:
    filtered = filtered[filtered['airline'] == sel_airline]
if sel_region:
    filtered = filtered[filtered['region'] == sel_region]

# KPI cards
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1,1,1])
c1.markdown(f"<div class='kpi-card'><b>✈️ Airline</b><div>{sel_airline}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi-card'><b>🌍 Region</b><div>{sel_region}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi-card'><b>📊 Rows (filtered)</b><div>{len(filtered)}</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.divider()
st.markdown("### 📈 Compensation Trends")
st.caption("Overview of historical compensation patterns across the dataset.")
st.markdown("<br>", unsafe_allow_html=True)
colA, colB = st.columns([2,1])

# Chart 1: avg compensation by ticket_class (filtered)
with colA:
    avg_class = filtered.groupby('ticket_class', as_index=False)['final_compensation'].mean().sort_values('final_compensation', ascending=False)
    fig1, ax1 = plt.subplots(figsize=(8,4))
    if not avg_class.empty:
        ax1.bar(avg_class['ticket_class'], avg_class['final_compensation'], color='#5ab4ac')
        ax1.set_title("Average Compensation by Ticket Class")
        ax1.set_ylabel("Compensation (₹)")
        ax1.set_xlabel("Ticket Class")
        ax1.tick_params(axis='x', rotation=0)
    else:
        ax1.text(0.5,0.5,"No data for selected filters", ha='center')
    st.pyplot(fig1)

# Chart 2: scatter delay vs compensation (filtered)
with colB:
    fig2, ax2 = plt.subplots(figsize=(5,4))
    if len(filtered) >= 3:
        ax2.scatter(filtered['delay_minutes'], filtered['final_compensation'], alpha=0.6)
        ax2.set_title("Delay vs Compensation")
        ax2.set_xlabel("Delay (min)")
        ax2.set_ylabel("Compensation (₹)")
    else:
        ax2.text(0.5,0.5,"Not enough filtered rows", ha='center')
    st.pyplot(fig2)

st.markdown("---")
st.divider()
st.markdown("### 🤖 Predict Compensation")
st.caption("AI-estimated compensation based on delay, distance, class, and loyalty.")
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

# ----------------------------
# Prediction only when button is clicked
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("💵 Predict Compensation"):
    # encode inputs
    air_enc = int(airline_encoder.transform([sel_airline])[0])
    ticket_enc = int(ticket_encoder.transform([sel_ticket])[0])
    region_enc = int(region_encoder.transform([sel_region])[0])

    X_new = np.array([[delay, distance, loyalty, air_enc, ticket_enc, region_enc]])
    pred = model.predict(X_new)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"💡 Estimated Compensation: ₹{pred:,.2f}")
    st.caption(f"Inputs — Delay: {delay} min • Distance: {distance} km • Loyalty: {loyalty:.2f}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------
    # Passenger Experience Indicator (STEP 5)
    # ----------------------------
    st.divider()
    st.markdown("### 🧍 Passenger Experience Indicator")
    st.caption("Passenger sentiment inferred from disruption severity.")
    if delay <= 60:
        st.success("🙂 Calm passenger — minimal disruption")
    elif 61 <= delay <= 120:
        st.warning("😐 Inconvenienced passenger — moderate delay")
    elif 121 <= delay <= 180:
        st.warning("😟 Stressed passenger — significant delay impact")
    else:
        st.error("😠 Highly dissatisfied passenger — severe disruption")
    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------
    # Feature Importance Visualization
    # ----------------------------
    st.markdown("<div class='feature-importance-container'>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<h3 style='color: #ffffff; margin-bottom: 16px;'>🔍 Explainable AI – Feature Importance</h3>", unsafe_allow_html=True)
    st.caption("Relative contribution of each factor to the model's decision.")
    st.markdown("<div class='feature-importance-text'>This chart shows which factors most influenced the compensation prediction.</div>", unsafe_allow_html=True)
    
    # Get feature importances
    feature_names = ['Delay (minutes)', 'Distance (km)', 'Loyalty Score', 'Airline', 'Ticket Class', 'Region']
    importances = model.feature_importances_
    
    # Create DataFrame and sort by importance (descending)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Create horizontal bar chart with dark theme
    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
    fig_importance.patch.set_facecolor('#0f172a')
    ax_importance.set_facecolor('#0f172a')
    
    bars = ax_importance.barh(importance_df['Feature'], importance_df['Importance'], 
                               color=(0.23, 0.51, 0.96, 0.8), edgecolor=(1.0, 1.0, 1.0, 0.1), linewidth=1)
    
    ax_importance.set_xlabel('Importance', color='#e2e8f0', fontsize=12, fontweight=500)
    ax_importance.set_title('Feature Importance Analysis', color='#ffffff', fontsize=16, fontweight=600, pad=20)
    ax_importance.tick_params(colors='#cbd5e1', labelsize=11)
    ax_importance.spines['top'].set_visible(False)
    ax_importance.spines['right'].set_visible(False)
    ax_importance.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.1))
    ax_importance.spines['left'].set_color((1.0, 1.0, 1.0, 0.1))
    ax_importance.grid(axis='x', alpha=0.1, color='#ffffff', linestyle='--')
    ax_importance.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax_importance.text(row['Importance'] + 0.01, i, f"{row['Importance']:.3f}", 
                          va='center', color='#e2e8f0', fontsize=10, fontweight=500)
    
    plt.tight_layout()
    st.pyplot(fig_importance)
    plt.close(fig_importance)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------
    # SHAP Explanation Visualization
    # ----------------------------
    if SHAP_AVAILABLE:
        try:
            st.markdown("<div class='feature-importance-container'>", unsafe_allow_html=True)
            st.divider()
            st.markdown("<h3 style='color: #ffffff; margin-bottom: 16px;'>🧠 Explainable AI – Why This Compensation Was Predicted</h3>", unsafe_allow_html=True)
            st.caption("Transparent breakdown of how this prediction was formed.")
            st.markdown("<div class='feature-importance-text'>This chart explains how each factor influenced the final compensation amount for this passenger.</div>", unsafe_allow_html=True)
            
            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_new)
            
            # Handle single output (regression) vs multiple outputs
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Create DataFrame with SHAP values
            feature_names = ['Delay (minutes)', 'Distance (km)', 'Loyalty Score', 'Airline', 'Ticket Class', 'Region']
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values
            })
            
            # Sort by absolute SHAP value (descending)
            shap_df['Abs_SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values('Abs_SHAP', ascending=False)
            
            # Create horizontal bar chart with dark theme
            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            fig_shap.patch.set_facecolor('#0f172a')
            ax_shap.set_facecolor('#0f172a')
            
            # Color bars: blue for positive, red for negative
            colors = [(0.23, 0.51, 0.96, 0.8) if val >= 0 else (0.96, 0.26, 0.21, 0.8) for val in shap_df['SHAP Value']]
            
            bars = ax_shap.barh(shap_df['Feature'], shap_df['SHAP Value'], 
                               color=colors, edgecolor=(1.0, 1.0, 1.0, 0.1), linewidth=1)
            
            ax_shap.set_xlabel('SHAP Value (Impact on Compensation)', color='#e2e8f0', fontsize=12, fontweight=500)
            ax_shap.set_title('SHAP Explanation: Feature Contributions', color='#ffffff', fontsize=16, fontweight=600, pad=20)
            ax_shap.axvline(x=0, color='#cbd5e1', linestyle='--', linewidth=1, alpha=0.5)
            ax_shap.tick_params(colors='#cbd5e1', labelsize=11)
            ax_shap.spines['top'].set_visible(False)
            ax_shap.spines['right'].set_visible(False)
            ax_shap.spines['bottom'].set_color((1.0, 1.0, 1.0, 0.1))
            ax_shap.spines['left'].set_color((1.0, 1.0, 1.0, 0.1))
            ax_shap.grid(axis='x', alpha=0.1, color='#ffffff', linestyle='--')
            ax_shap.set_axisbelow(True)
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(shap_df.iterrows()):
                value = row['SHAP Value']
                ax_shap.text(value + (0.01 if value >= 0 else -0.01), i, f"{value:.3f}", 
                           va='center', ha='left' if value >= 0 else 'right', 
                           color='#e2e8f0', fontsize=10, fontweight=500)
            
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"⚠️ SHAP explanation unavailable: {str(e)}")
    else:
        # Production-friendly message: do not suggest runtime installs on Streamlit Cloud
        st.info("ℹ️ SHAP-based explainability is unavailable in this deployment. The app will continue without SHAP visualizations.")

    # PDF summary generation
    def make_pdf_bytes(airline, ticket, region, delay, distance, loyalty, predicted):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 820, "Dynamic Travel Disruption — Compensation Summary")
        c.setFont("Helvetica", 11)
        y = 780
        c.drawString(50, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 30
        c.drawString(50, y, f"Airline: {airline}")
        y -= 20
        c.drawString(50, y, f"Ticket Class: {ticket}")
        y -= 20
        c.drawString(50, y, f"Region: {region}")
        y -= 20
        c.drawString(50, y, f"Delay (minutes): {delay}")
        y -= 20
        c.drawString(50, y, f"Distance (km): {distance}")
        y -= 20
        c.drawString(50, y, f"Loyalty score: {loyalty:.2f}")
        y -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Predicted compensation: ₹{pred:,.2f}")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.read()

    pdf_bytes = make_pdf_bytes(sel_airline, sel_ticket, sel_region, delay, distance, loyalty, pred)
    st.download_button("📄 Download Summary PDF", data=pdf_bytes, file_name="compensation_summary.pdf", mime="application/pdf")
    st.caption("Download a printable summary of inputs and the predicted compensation.")
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
cols = st.columns([1,2,1])
cols[1].caption("🚀 Developed by Riya Vaishya | B.Sc. Data Science – Final Year Project")