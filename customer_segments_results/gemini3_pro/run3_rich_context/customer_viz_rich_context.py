'''
You are an expert data scientist and visualization specialist.

## Dataset Context:
The Customer Segments dataset contains marketing and purchasing behavior data for customers.

## Dataset Schema (subset used for clustering):
Use ONLY the following numeric features:
- Income
- Recency
- MntWines
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds
- NumDealsPurchases
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases
- NumWebVisitsMonth
- Kidhome
- Teenhome

## Technical Requirements:
- Use Python with pandas, matplotlib, seaborn, or plotly
- Load data from 'data/customer_segments.csv'
- Use ONLY the specified features
- Standardize the features
- Reduce to 2D using PCA
- Perform k-means clustering with k=3
- Save the final visualization as 'output.png' with DPI=300
- Include all necessary imports
- Make the code fully executable

## Design Principles:
- Use readable axis labels and title
- Use colorblind-friendly cluster colors
- Include legend

## Your Task:
Create a 2D visualization that clusters customers based on their purchasing behavior.

Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# Set visual style
sns.set_theme(style="whitegrid")

# Load data (handling both requested path and typical local path)
file_path = 'data/customer_segments.csv' if os.path.exists('data/customer_segments.csv') else 'customer_segments.csv'
df = pd.read_csv(file_path)

# Extract specified features
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'Kidhome', 'Teenhome'
]

# Subset and handle missing values
data = df[features].dropna()

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Reduce to 2D using PCA
pca = PCA(n_components=2, random_state=42)
data_pca = pca.fit_transform(data_scaled)

# Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(data_pca)

# Create visualization
plt.figure(figsize=(10, 8))

# Scatter plot with colorblind-friendly palette
sns.scatterplot(
    x=data_pca[:, 0], 
    y=data_pca[:, 1], 
    hue=cluster_labels, 
    palette='colorblind', 
    s=70, 
    alpha=0.8,
    edgecolor='w',
    linewidth=0.5
)

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0], 
    centers[:, 1], 
    c='black', 
    s=200, 
    marker='X', 
    label='Centroids'
)

# Format axes, title, and legend
plt.title('Customer Segments based on Purchasing Behavior\n(PCA Reduced to 2D, k=3)', fontsize=16, pad=15)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)

# Update legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Centroids'], title='Customer Groups', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Save final visualization
plt.savefig('customer_segments_results/gemini3_pro/run3_rich_context/customer_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()