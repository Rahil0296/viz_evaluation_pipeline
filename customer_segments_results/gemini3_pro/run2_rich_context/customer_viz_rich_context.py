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

# Load data (handling both requested and actual file locations)
file_path = 'data/customer_segments.csv' if os.path.exists('data/customer_segments.csv') else 'customer_segments.csv'
df = pd.read_csv(file_path)

# Specify the features to use
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth', 'Kidhome', 'Teenhome'
]

# Subset and drop missing values (e.g., in Income)
X = df[features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Create a DataFrame for visualization
plot_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Cluster': ['Cluster ' + str(c) for c in clusters]
})

# Configure plot aesthetics
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# Use a colorblind-friendly palette
cb_palette = sns.color_palette("colorblind", 3)

# Create the scatter plot
sns.scatterplot(
    data=plot_df,
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    palette=cb_palette,
    alpha=0.8,
    s=70,
    edgecolor='w'
)

# Set readable labels and title
plt.title('Customer Segments clustered by Purchasing Behavior', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Principal Component 1', fontsize=12, fontweight='bold')
plt.ylabel('Principal Component 2', fontsize=12, fontweight='bold')

# Customize legend
plt.legend(title='Customer Groups', title_fontsize='12', fontsize='11', 
           loc='upper right', frameon=True, shadow=True)

# Save the visualization
plt.tight_layout()

plt.savefig('customer_segments_results/gemini3_pro/run2_rich_context/customer_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()