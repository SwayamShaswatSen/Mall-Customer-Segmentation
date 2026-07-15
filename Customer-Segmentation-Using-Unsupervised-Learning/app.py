import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mall Customer Analytics",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Mall_Customers.csv")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛍️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📂 Customer Data",
        "📊 Visual Analysis",
        "🤖 Customer Clusters",
        "💡 Insights",
        "⬇ Export",
        "ℹ About"
    ]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "🏠 Home":

    st.title("🛍️ Mall Customer Analytics Dashboard")
    st.write("### Internship Project")
    st.write(
        "This dashboard uses the K-Means Clustering algorithm "
        "to group customers according to their Annual Income and Spending Score."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Customers", len(df))

    with col2:
        st.metric("Clusters", 5)

    with col3:
        st.metric(
            "Average Income",
            f"${df['Annual Income (k$)'].mean():.1f}k"
        )

    st.divider()

    st.subheader("Project Overview")

    st.info("""
This project demonstrates customer segmentation using the K-Means clustering algorithm.

The dashboard allows us to:

• View customer information

• Visualize spending patterns

• Create customer clusters

• Understand business insights
""")
    # -----------------------------
# Customer Data Page
# -----------------------------
elif page == "📂 Customer Data":

    st.title("📂 Customer Dataset")

    st.write("Preview of the mall customer dataset.")

    st.dataframe(df)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")

    with col2:
        st.write("**Columns Available:**")
        st.write(list(df.columns))


# -----------------------------
# Visual Analysis Page
# -----------------------------
elif page == "📊 Visual Analysis":

    st.title("📊 Customer Distribution")

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Gender",
        size="Age",
        hover_data=["CustomerID"],
        title="Income vs Spending Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age Distribution")

    fig2 = px.histogram(
        df,
        x="Age",
        nbins=20,
        color="Gender",
        title="Customer Age Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)
    # -----------------------------
# Customer Clusters Page
# -----------------------------
elif page == "🤖 Customer Clusters":

    st.title("🤖 K-Means Customer Segmentation")

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X)

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        title="Customer Segments",
        labels={"color": "Cluster"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")

    summary = df.groupby("Cluster").agg({
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean"
    }).round(2)

    st.dataframe(summary, use_container_width=True)
    # -----------------------------
# Business Insights Page
# -----------------------------
elif page == "💡 Insights":

    st.title("💡 Business Insights")

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X)

    summary = df.groupby("Cluster").agg({
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean"
    }).round(2)

    st.dataframe(summary, use_container_width=True)

    st.markdown("### Recommendations")

    st.success("""
**Cluster 0:** Focus on loyalty rewards.

**Cluster 1:** Offer premium products.

**Cluster 2:** Target with discounts.

**Cluster 3:** Improve engagement through marketing.

**Cluster 4:** Maintain customer satisfaction with personalized offers.
""")

# -----------------------------
# Export Page
# -----------------------------
elif page == "⬇ Export":

    st.title("⬇ Export Data")

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Clustered Dataset",
        data=csv,
        file_name="Mall_Customers_Segmented.csv",
        mime="text/csv"
    )

# -----------------------------
# About Page
# -----------------------------
elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
### Mall Customer Segmentation using K-Means

**Project Type:** Internship Project

**Machine Learning:** Unsupervised Learning

**Algorithm:** K-Means Clustering

**Dataset:** Mall Customers Dataset

**Tools Used:**
- Python
- Pandas
- Plotly
- Scikit-Learn
- Streamlit

---

**Developed by:**  
**Swayam Shaswat Sen**
""")
