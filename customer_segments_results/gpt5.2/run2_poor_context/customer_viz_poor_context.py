'''
POOR CONTEXT PROMPT USED:

The Customer Segments dataset contains marketing-related information about customers.
Create a visualization that shows the main clusters or patterns in the data.
Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data/customer_segments.csv")

# Select numeric features
numeric_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['ID'], errors='ignore')

# Handle missing values
numeric_df = numeric_df.fillna(numeric_df.median())

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=clusters)
plt.title("Customer Segments Clusters (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.savefig('customer_segments_results/gpt5.2/run2_poor_context/customer_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()