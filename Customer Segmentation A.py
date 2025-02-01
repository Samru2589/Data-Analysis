# 1) Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2) Load the dataset
data = pd.read_csv(r"C:\Users\Samruddhi Yadav\Documents\Data Science M\projects Resume\Customer Segmentation Analysis\customer_segmentation_data.csv")

# Display the first few rows
print(data.head())

# 3) Explore and Clean the Data
# Check for missing values
print(data.isnull().sum())

# There are no missing values

# Summary statistics
print(data.describe())

# Visualize distributions
sns.pairplot(data)
plt.show()

# 4)Feature Scaling
# Select relevant features for clustering
features = ['Annual_Income', 'Spending_Score']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display scaled data
print(X_scaled[:5])

# 5) Determine Optimal Number of Clusters (Elbow Method)
# Use the elbow method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# 6) Apply K-means with the optimal number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Display cluster counts
print(data['Cluster'].value_counts())

# 7) Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segments')
plt.show()

# 8) Analyze the characteristics of each cluster
for cluster in sorted(data['Cluster'].unique()):
    print(f"Cluster {cluster} Summary:")
    print(data[data['Cluster'] == cluster][features].mean())
    print("\n")
    
    
# 9) Export Results
# Save the results to a CSV file
data.to_csv('customer_segments.csv', index=False)

# 10) Deploy as a Web Application (Using Streamlit)

import streamlit as st

# Load data
st.title("Customer Segmentation Analysis")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Feature Selection
    features = ['Annual_Income', 'Spending_Score']
    if all(feature in data.columns for feature in features):
        X = data[features]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Optimal number of clusters (Elbow Method)
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        # Elbow Curve
        st.subheader("Elbow Curve")
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), inertia, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        st.pyplot(fig)

        # K-means Clustering
        st.subheader("K-means Clustering")
        optimal_clusters = st.slider("Select the number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        data['Cluster'] = clusters
        st.write("Clustered Data Preview:", data.head())

        # Visualize Clusters
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x='Annual_Income', y='Spending_Score', hue='Cluster', data=data, palette='viridis', ax=ax
        )
        ax.set_title('Customer Segments')
        st.pyplot(fig)

        # Download Clustered Data
        st.subheader("Download Results")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Clustered Data as CSV",
            data=csv,
            file_name='customer_segments.csv',
            mime='text/csv',
        )
    else:
        st.error("The dataset must contain 'Annual_Income' and 'Spending_Score' columns.")
