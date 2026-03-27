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

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('data/customer_segments.csv')

# -----------------------------
# Select Specified Features
# -----------------------------
features = [
    "Income",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Kidhome",
    "Teenhome"
]

# FIX : .dropna()  Model did not handle missing values.
X = df[features].dropna().copy()

# -----------------------------
# Standardize Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA Reduction to 2D
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# -----------------------------
# K-Means Clustering (k=3)
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

pca_df["Cluster"] = clusters

# -----------------------------
# Visualization
# -----------------------------
sns.set(style="whitegrid")
palette = sns.color_palette("colorblind", 3)

plt.figure(figsize=(10, 7))

for cluster_id in sorted(pca_df["Cluster"].unique()):
    subset = pca_df[pca_df["Cluster"] == cluster_id]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=40,
        color=palette[cluster_id],
        label=f"Cluster {cluster_id}",
        alpha=0.8
    )

plt.title("Customer Segments Based on Purchasing Behavior (K-Means, k=3)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Clusters")

plt.tight_layout()

# -----------------------------
# Save Output
# -----------------------------
plt.savefig('customer_segments_results/gpt5.2/run2_rich_context/customer_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()