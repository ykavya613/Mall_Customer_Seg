# 🛍️ Mall Customer Segmentation using Machine Learning

This project performs customer segmentation for a mall business using unsupervised learning techniques. By analyzing purchasing patterns and demographic data, it helps the business better understand its customer base and target marketing strategies.

---

## 🎓 Internship Details

- **Program:** Summer Internship program 2025
- **Organizer:** Bharatversity
- **Duration:** 7 weeks
- Domain : Data Science
- **Project Title:** Mall Customer Segmentation using Machine Learning
- **Intern:** yelugu kavya



---

## 📁 Dataset Overview

- **Dataset:** `Mall_Customers.csv`
- **Features Used:**
  - CustomerID
  - Gender
  - Age
  - Annual Income (k\$)
  - Spending Score (1–100)

---

## 📊 Exploratory Data Analysis (EDA)

### 🔍 Step-by-Step Visual Insights:

  I have explored visually:

- **Data Cleaning:** Checked for null values, ensured correct data types, and removed duplicates if any.

- **Gender Distribution:**

  - **Visualization:** Bar chart showing count of male vs. female customers.
  - **Insight:** Helps understand the gender balance in the customer base.

- **Age Distribution:**

  - **Visualization:** Histogram of ages.
  - **Insight:** Identified which age groups frequent the mall the most (e.g., most customers were aged between 30–40).

- **Annual Income Histogram:**

  - **Visualization:** Histogram for annual income.
  - **Insight:** Highlighted common income brackets and potential buying power.

- **Spending Score Histogram:**

  - **Visualization:** Histogram of spending scores (1–100).
  - **Insight:** Revealed the spread of customer purchasing activity from low to high spenders.

- **Income vs Spending Score Scatter Plot:**

  - **Visualization:** 2D scatter to find natural groupings.
  - **Insight:** Showed patterns indicating segments like high income–low spenders or moderate income–high spenders.

- **Age vs Spending Score Scatter Plot:**

  - **Visualization:** Scatter plot of age vs. spending score.
  - **Insight:** Helped correlate age and spending behavior for targeted promotions.
  - Gender Distribution
  - Age Distribution
  - Annual Income and Spending Score histograms
  - Scatter plots to visualize clustering potential

---

## 📈 Unsupervised Learning - Clustering

Applied **K-Means Clustering** to segment customers:

- **Feature Selection:** Focused on Age, Annual Income, and Spending Score.
- **Elbow Method:** Used to determine optimal number of clusters (k).
- **Model Training:** Implemented K-Means with identified value of k.
- **Cluster Labeling:** Added cluster labels to customer data.

### 📌 Cluster Analysis

Visualized cluster segments using 2D and 3D scatter plots:

- Identified unique customer segments based on income and spending patterns.
- Made observations on high-value customers, budget-conscious customers, etc.

---

## 🚀 Deployment with Streamlit

### 🧰 To Deploy:

1. Install Streamlit if not already:

```bash
pip install streamlit
```

2. Create a Python script (e.g., `app.py`) with these components:

- File uploader for CSV
- Data display (using Pandas)
- Plotly/Seaborn visualizations
- K-Means clustering with slider input for number of clusters
- Display final cluster assignments and plots

3. Run the app locally:

```bash
streamlit run app.py
```

### 🔗 Deployment (Cloud):

- **Streamlit Community Cloud** or **Render** can be used to deploy your app publicly.



---

## 🔧 Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Streamlit (for deployment)

---

## 📌 Conclusion

This project demonstrates how unsupervised learning can help segment customers in a retail setting. Through K-Means clustering and visualization, we gain insights into customer behavior which can directly inform marketing and sales strategies.

The internship improved my ability to:

- Apply unsupervised ML techniques to real-world data
- Perform detailed data visualization and interpretation
- Deploy ML models using Streamlit for interactive use

---

## &#x20;&#x20;

