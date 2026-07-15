import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mall Customer Analytics",
    page_icon="🛍️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main{
    background:#0E1117;
}

h1{
    color:white;
}

.stMetric{
    background:#1E1E1E;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 0px 20px rgba(0,255,255,.15);
}

div[data-testid="stDataFrame"]{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛍️ Mall Customer Analytics Dashboard")
st.markdown("### AI-Powered Customer Segmentation using K-Means")
st.markdown("---")

# ---------------- LOAD DATA ----------------
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Mall_Customers.csv")

# ---------------- DASHBOARD METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Customers", len(df))

with col2:
    st.metric("💰 Avg Income", f"{df['Annual Income (k$)'].mean():.1f} K")

with col3:
    st.metric("🎯 Avg Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

with col4:
    st.metric("🧑 Avg Age", f"{df['Age'].mean():.1f}")

# ---------------- DATASET ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------- CUSTOMER DISTRIBUTION ----------------
st.subheader("📊 Customer Distribution")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Gender",
    size="Age",
    title="Mall Customers Overview"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- K-MEANS ----------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# ---------------- CLUSTER GRAPH ----------------
st.subheader("🎯 Customer Segments")

fig2 = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=df["Cluster"].astype(str),
    title="Customer Segmentation using K-Means"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- SUMMARY ----------------
st.subheader("📋 Cluster Summary")
st.dataframe(df.groupby("Cluster").mean(numeric_only=True), use_container_width=True)

# ---------------- SUCCESS ----------------
st.success("✅ Project Completed Successfully!")
