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

# Load data (handling potential path variations)
try:
    df = pd.read_csv('data/customer_segments.csv')
except FileNotFoundError:
    df = pd.read_csv('customer_segments.csv')

# Define the specified numeric features
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth', 'Kidhome', 'Teenhome'
]

# Subset the dataset to use ONLY the specified features and drop missing values (e.g., Income)
X = df[features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Principal Component 1': X_pca[:, 0],
    'Principal Component 2': X_pca[:, 1],
    'Cluster': cluster_labels
})

# Set up the visualization
plt.figure(figsize=(10, 7))

# Use a colorblind-friendly palette
cb_palette = sns.color_palette("colorblind", 3)

# Create the scatter plot
sns.scatterplot(
    data=plot_df,
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Cluster',
    palette=cb_palette,
    alpha=0.8,
    s=60,
    edgecolor='w',
    linewidth=0.5
)

# Apply design principles: readable axis labels, title, and legend
plt.title('Customer Segments Clustering (2D PCA Projection)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Format the legend
plt.legend(title='Customer Cluster', title_fontsize=12, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save the final visualization
plt.savefig('customer_segments_results/gemini3_pro/run2_rich_context/customer_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()
