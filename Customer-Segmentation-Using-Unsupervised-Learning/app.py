import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Mall Customer Analytics",
    page_icon="🛍️",
    layout="wide"
)

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

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation")
st.write("### Internship Project")
st.write("This project uses K-Means Clustering to segment mall customers based on Annual Income and Spending Score.")

# Load dataset
from pathlib import Path

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Scatter plot
st.subheader("Customer Distribution")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Gender",
    size="Age",
    title="Mall Customers Overview"
)

st.plotly_chart(fig, use_container_width=True)

# K-Means
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

st.subheader("Customer Segments")

fig2 = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=df["Cluster"].astype(str),
    title="Customer Segmentation using K-Means"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Cluster Summary")
st.dataframe(df.groupby("Cluster").mean(numeric_only=True))

st.success("Project Completed Successfully ✅")
