import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from PIL import Image
import plotly.express as px

# Page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# Background CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1536520002442-39764a41e1f4");
        background-size: cover;
        background-position: center;
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #ffdd57;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="main-title">🛍️ Mall Customer Segmentation App</div>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

    # Feature selection
    df_model = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # EDA Visuals
    st.subheader("📊 Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="Age", title="Age Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Annual Income (k$)", title="Annual Income Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(df, x="Spending Score (1-100)", title="Spending Score Distribution")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)", color="Gender", title="Income vs Spending Score")
        st.plotly_chart(fig4, use_container_width=True)

    # Clustering
    st.subheader("🔀 Customer Segmentation")
    k = st.slider("Select number of clusters (k)", 2, 10, 5)
    model = KMeans(n_clusters=k)
    df_model['Cluster'] = model.fit_predict(df_model)

    fig5 = px.scatter_3d(
        df_model, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
        color='Cluster', title=f"3D Cluster Visualization (k={k})",
        color_continuous_scale='Rainbow')
    st.plotly_chart(fig5, use_container_width=True)

    st.success("Segmentation complete. You can analyze customer clusters for business insights!")

else:
    st.info("Please upload a CSV file to begin.")

