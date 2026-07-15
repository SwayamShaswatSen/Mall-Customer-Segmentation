import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from pathlib import Path

st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Mall_Customers.csv")

st.sidebar.title("📊 Retail Intelligence")

page = st.sidebar.selectbox(
    "Navigate",
    (
        "Dashboard",
        "Customer Explorer",
        "Spending Analytics",
        "Smart Segmentation",
        "Marketing Suggestions",
        "Export Report",
        "About"
    )
)
# ===========================
# DASHBOARD
# ===========================

if page == "Dashboard":

    st.title("📊 Retail Customer Intelligence")

    st.write(
        "Analyze customer behaviour using Machine Learning and K-Means Clustering."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        len(df)
    )

    c2.metric(
        "Average Age",
        round(df["Age"].mean(),1)
    )

    c3.metric(
        "Average Income",
        round(df["Annual Income (k$)"].mean(),1)
    )

    c4.metric(
        "Average Spending",
        round(df["Spending Score (1-100)"].mean(),1)
    )

    st.divider()

    st.subheader("Dataset Snapshot")

    st.dataframe(df.head(10),use_container_width=True)

    st.info(
        "Use the menu on the left to explore customer information, analytics and segmentation."
    )

# ===========================
# CUSTOMER EXPLORER
# ===========================

elif page == "Customer Explorer":

    st.title("👥 Customer Explorer")

    customer = st.slider(
        "Choose Customer ID",
        1,
        len(df),
        1
    )

    row = df[df["CustomerID"]==customer]

    st.write(row)

    st.subheader("Customer Details")

    st.write(
        f"Gender : {row.iloc[0]['Gender']}"
    )

    st.write(
        f"Age : {row.iloc[0]['Age']}"
    )

    st.write(
        f"Annual Income : {row.iloc[0]['Annual Income (k$)']} K$"
    )

    st.write(
        f"Spending Score : {row.iloc[0]['Spending Score (1-100)']}"
    )
    # ===========================
# SPENDING ANALYTICS
# ===========================

elif page == "Spending Analytics":

    st.title("📈 Spending Analytics")

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.histogram(
            df,
            x="Age",
            nbins=20,
            title="Age Distribution",
            color="Gender"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        fig2 = px.histogram(
            df,
            x="Annual Income (k$)",
            nbins=20,
            title="Income Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    fig3 = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Gender",
        size="Age",
        hover_data=["CustomerID"],
        title="Income vs Spending Score"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Quick Observations")

    st.markdown("""
- Younger customers generally show higher spending.
- Spending varies even among customers with similar income.
- Customer behaviour cannot be explained using income alone.
""")
    # ===========================
# SMART SEGMENTATION
# ===========================

elif page == "Smart Segmentation":

    st.title("🧠 Smart Customer Segmentation")

    clusters = st.slider(
        "Select Number of Customer Segments",
        min_value=2,
        max_value=10,
        value=5
    )

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    model = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X)

    cluster_names = {
        0: "💎 Premium Customers",
        1: "🛍 Regular Customers",
        2: "💰 High Income - Low Spending",
        3: "🎯 Budget Customers",
        4: "⭐ High Spending Customers"
    }

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        hover_data=["CustomerID", "Age", "Gender"],
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")

    summary = df.groupby("Cluster")[
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    ].mean().round(2)

    st.dataframe(summary, use_container_width=True)

    st.subheader("Segment Description")

    for cluster in sorted(df["Cluster"].unique()):

        st.markdown(f"### {cluster_names.get(cluster, f'Cluster {cluster}')}")

        age = summary.loc[cluster, "Age"]
        income = summary.loc[cluster, "Annual Income (k$)"]
        spending = summary.loc[cluster, "Spending Score (1-100)"]

        st.write(f"Average Age : {age}")
        st.write(f"Average Income : {income} K$")
        st.write(f"Average Spending Score : {spending}")

        if income > 60 and spending > 60:
            st.success("Marketing Strategy: Offer premium memberships and exclusive rewards.")

        elif income > 60 and spending <= 60:
            st.info("Marketing Strategy: Encourage spending with personalized offers.")

        elif income <= 60 and spending > 60:
            st.warning("Marketing Strategy: Promote loyalty programs and discounts.")

        else:
            st.error("Marketing Strategy: Focus on budget-friendly campaigns.")
# ===========================
# MARKETING SUGGESTIONS
# ===========================

elif page == "Marketing Suggestions":

    st.title("🎯 Marketing Suggestions")

    st.markdown("""
### Suggested Strategies

💎 **Premium Customers**
- Offer VIP memberships
- Exclusive product launches
- Premium rewards

🛍 **Regular Customers**
- Seasonal discounts
- Cashback offers
- Bundle deals

💰 **High Income - Low Spending**
- Personalized recommendations
- Premium advertisements
- Loyalty campaigns

🎯 **Budget Customers**
- Affordable products
- Discount coupons
- Festival sales

⭐ **High Spending Customers**
- Reward points
- Early access to sales
- Referral bonuses
""")

# ===========================
# EXPORT REPORT
# ===========================

elif page == "Export Report":

    st.title("📥 Export Report")

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

# ===========================
# ABOUT
# ===========================

elif page == "About":

    st.title("ℹ About")

    st.markdown("""
## Retail Customer Intelligence

This project demonstrates **Customer Segmentation using the K-Means Clustering Algorithm**.

### Technologies Used

- Python
- Pandas
- Plotly
- Scikit-Learn
- Streamlit

### Machine Learning Technique

- Unsupervised Learning
- K-Means Clustering

### Dataset

Mall Customers Dataset

---

### Developed By

**Swayam Shaswat Sen**

Internship Project
""")

st.markdown("---")
st.caption("© 2026 Retail Customer Intelligence Dashboard")
