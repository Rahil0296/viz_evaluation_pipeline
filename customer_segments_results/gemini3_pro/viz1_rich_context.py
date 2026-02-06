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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set visualization style
sns.set(style="whitegrid")

# Load data
df = pd.read_csv('data/customer_segments.csv')

# Define features
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'Kidhome', 'Teenhome'
]

# Select specified features
data = df[features]

# Handle missing values
data = data.dropna()

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# PCA to 2D
pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(scaled_data)

# K-Means on PCA space
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_data)

# Prepare DataFrame for plotting
plot_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
plot_df['Cluster'] = clusters.astype(str)

# Visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=plot_df,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='viridis',
    s=60,
    alpha=0.8,
    edgecolor='w'
)

plt.title('Customer Segments Clustering (PCA Projection)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('customer_segments_results/gemini3_pro/viz1_rich_context_output.png', dpi=300)
plt.close()
