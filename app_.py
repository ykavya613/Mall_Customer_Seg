 # app.py

# 📦 Import Required Libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# 🖼️ Streamlit Page Setup
st.set_page_config(page_title="🛍️ Mall Customer Segmentation", layout="wide")

# 🎨 Light Green Themed Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f7f7; /* Light gray background */
        color: #66cc66 !important; /* Light green text */
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #66cc66;
    }
    h1, h2, h3, h4, h5, h6, .css-1v0mbdj p {
        color: #66cc66 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🏷️ App Title
st.markdown('<div class="main-title">🛍️ Mall Customer Segmentation</div>', unsafe_allow_html=True)

# 📂 Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# 👀 Dataset Preview
st.subheader("📄 Preview of Customer Data")
st.write("Here's a sample of the data we collected from mall customers:")
st.dataframe(df.head())

# 🎯 Features for Clustering
st.subheader("🔍 Features Used for Clustering")
st.write("We use Age, Annual Income, and Spending Score to group similar customers.")
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df_model = df[features].copy()

# 📊 Visualizing Data
st.subheader("📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.histogram(df, x='Age', title="Age Distribution", color_discrete_sequence=['#aaaaaa']), use_container_width=True)
    st.plotly_chart(px.histogram(df, x='Annual Income (k$)', title="Annual Income Distribution", color_discrete_sequence=['#aaaaaa']), use_container_width=True)

with col2:
    st.plotly_chart(px.histogram(df, x='Spending Score (1-100)', title="Spending Score Distribution", color_discrete_sequence=['#aaaaaa']), use_container_width=True)
    st.plotly_chart(px.scatter(
        df, x='Annual Income (k$)', y='Spending Score (1-100)',
        color='Gender', title="Income vs Spending Score by Gender",
        color_discrete_map={"Male": "#66cc66", "Female": "#aaaaaa"}
    ), use_container_width=True)

# 🔀 Clustering Section
st.subheader("🔀 Segmenting Customers Using K-Means Clustering")
st.write("You can choose how many groups (clusters) you'd like to divide the customers into.")

k = st.slider("Select number of customer groups (clusters)", min_value=2, max_value=10, value=5)

# ⚙️ Apply Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df_model['Cluster'] = kmeans.fit_predict(df_model)

# 📈 3D Cluster Visualization
fig = px.scatter_3d(
    df_model, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
    color='Cluster', title=f"3D View of Customer Groups (k = {k})",
    color_continuous_scale='greens'
)
st.plotly_chart(fig, use_container_width=True)

# 🧮 Input vs Output Bar
st.subheader("🧮 Input vs Output Summary")

with st.expander("ℹ️ See how the data is processed"):
    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown("### 🔢 Input Features")
        st.write("""
        - **Age** (in years)  
        - **Annual Income (k$)** (thousands of dollars)  
        - **Spending Score (1-100)** (customer behavior score)  
        """)

    with col_out:
        st.markdown("### 🧠 Output")
        st.write(f"""
        - **Customer Segments**: {k} groups  
        - **Cluster Labels** added to data  
        - 3D visualization showing each group clearly  
        """)

# 📌 Final Business Insights
st.subheader("📌 Key Insights")
st.write("""
Based on clustering, we can identify groups like:

- 💰 **High-income, high spenders** – Premium customers  
- 👛 **Moderate spenders** – Price-conscious segment  
- 🧒 **Young spenders** – Likely to be impulsive  
- 🧓 **Older customers, low spend** – Conservative buyers  
- 🔍 **Other groups** – Depending on cluster number

Use this to:
- 🎯 Target your promotions better  
- 📦 Personalize offers  
- 💌 Improve customer loyalty  
""")
