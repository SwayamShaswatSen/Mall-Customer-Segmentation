import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from pathlib import Path

# ---------------- PAGE ---------------- #

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="🛍",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#09090B;
    color:white;
    font-family:Arial;
}

.main{
    background:#09090B;
}

.block-container{
    padding-top:2rem;
}

h1{
    text-align:center;
    color:#00F5FF;
    font-size:52px;
}

h2,h3{
    color:white;
}

div[data-testid="stMetric"]{

    background:#111827;

    border-radius:18px;

    padding:18px;

    border:1px solid #1F2937;

    box-shadow:0px 0px 25px rgba(0,255,255,.12);

}

.sidebar .sidebar-content{
    background:#111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #

st.markdown("""
# 🛍 Customer Analytics Dashboard

### AI Powered Customer Segmentation using K-Means

---
""")

# ---------------- LOAD DATA ---------------- #

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR/"Mall_Customers.csv")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("⚙ Dashboard")

clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    5
)

show_data = st.sidebar.checkbox(
    "Show Dataset",
    True
)

# ---------------- METRICS ---------------- #

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "👥 Customers",
        len(df)
    )

with c2:
    st.metric(
        "💰 Avg Income",
        f"{df['Annual Income (k$)'].mean():.1f} K"
    )

with c3:
    st.metric(
        "🎯 Avg Spending",
        f"{df['Spending Score (1-100)'].mean():.1f}"
    )

with c4:
    st.metric(
        "🧑 Avg Age",
        f"{df['Age'].mean():.1f}"
    )

st.markdown("---")
