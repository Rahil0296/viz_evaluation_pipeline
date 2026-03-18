'''
POOR CONTEXT PROMPT USED:

The Customer Segments dataset contains marketing-related information about customers.
Create a visualization that shows the main clusters or patterns in the data.
Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('data/customer_segments.csv')

# Drop irrelevant columns or columns that are not directly useful for distance-based clustering
cols_to_drop = ['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
df_features = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Handle categorical variables using one-hot encoding
df_features = pd.get_dummies(df_features, columns=['Education', 'Marital_Status'], drop_first=True)

# Impute missing values (e.g., Income)
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(df_features)

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Apply PCA for dimensionality reduction to 2 components for visualization
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(features_scaled)

# Apply KMeans clustering to find main patterns (using 4 clusters as an example)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'PCA Component 1': pca_components[:, 0],
    'PCA Component 2': pca_components[:, 1],
    'Cluster': clusters
})

# Create the visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=plot_df, 
    x='PCA Component 1', 
    y='PCA Component 2', 
    hue='Cluster', 
    palette='viridis', 
    alpha=0.7,
    s=50
)
plt.title('Customer Segments: Main Clusters Visualized via PCA', fontsize=16)
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('customer_segments_results/gemini3_pro/run2_poor_context/customer_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()

