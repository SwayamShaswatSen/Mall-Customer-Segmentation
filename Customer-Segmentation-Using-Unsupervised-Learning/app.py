import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from pathlib import Path

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍 Mall Customer Segmentation")
st.write("### Internship Project - Unsupervised Learning using K-Means")

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Dataset Information")

col1,col2,col3=st.columns(3)

col1.metric("Customers",len(df))
col2.metric("Average Income",round(df["Annual Income (k$)"].mean(),1))
col3.metric("Average Spending",round(df["Spending Score (1-100)"].mean(),1))

st.divider()

st.sidebar.header("Settings")

clusters=st.sidebar.slider(
    "Select Number of Clusters",
    2,
    10,
    5
)

X=df[["Annual Income (k$)","Spending Score (1-100)"]]

kmeans=KMeans(
    n_clusters=clusters,
    random_state=42,
    n_init=10
)

df["Cluster"]=kmeans.fit_predict(X)

st.subheader("Customer Segmentation")

fig=px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=df["Cluster"].astype(str),
    hover_data=["CustomerID","Age","Gender"],
    title="K-Means Customer Segmentation"
)

st.plotly_chart(fig,use_container_width=True)

st.subheader("Cluster Summary")

summary=df.groupby("Cluster")[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean()

st.dataframe(summary)

st.subheader("Business Insights")

for i in summary.index:

    st.write(f"### Cluster {i}")

    st.write(
        f"""
Average Age : {summary.loc[i,'Age']:.1f}

Average Income : {summary.loc[i,'Annual Income (k$)']:.1f} K$

Average Spending : {summary.loc[i,'Spending Score (1-100)']:.1f}
"""
    )

st.download_button(
    "Download Clustered Dataset",
    df.to_csv(index=False),
    "Mall_Customers_Segmented.csv"
)

st.success("Project Completed Successfully")
