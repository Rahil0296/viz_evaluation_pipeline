'''
RICH CONTEXT PROMPT USED:

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
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('data/customer_segments.csv')

# Selected numeric features
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'Kidhome', 'Teenhome'
]

X = df[features]

# FIX 1 (Execution Reliability): drop rows with missing values required by PCA
X = X.dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Create DataFrame for plotting
plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['Cluster'] = clusters

# Colorblind-friendly palette (Okabe-Ito)
colors = ['#0072B2', '#E69F00', '#009E73']

# Plot
plt.figure(figsize=(8, 6))
for cluster_id in sorted(plot_df['Cluster'].unique()):
    subset = plot_df[plot_df['Cluster'] == cluster_id]
    plt.scatter(
        subset['PC1'],
        subset['PC2'],
        label=f'Cluster {cluster_id + 1}',
        color=colors[cluster_id],
        alpha=0.7
    )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Segments (PCA + KMeans, k=3)')
plt.legend()
plt.tight_layout()

# FIX 2 (Reproducibility): save instead of show
plt.savefig('customer_segments_results/gpt5.2/viz1_rich_context_output.png', dpi=300, bbox_inches="tight")
plt.close()
