'''
POOR CONTEXT PROMPT USED:

The Customer Segments dataset contains marketing-related information about customers.
Create a visualization that shows the main clusters or patterns in the data.
Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('data/customer_segments.csv')

# Select relevant numerical columns for pattern extraction
features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
            'NumStorePurchases', 'NumWebVisitsMonth']

# Drop rows with missing values
data_clean = df[features].dropna()

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_clean)

# Apply K-Means clustering (using 4 clusters as a starting point)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame for plotting
plot_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
plot_df['Cluster'] = clusters

# Create the scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Cluster', 
    palette='viridis', 
    data=plot_df, 
    alpha=0.7, 
    s=70
)

# Enhance plot aesthetics
plt.title('Customer Segments Clusters (PCA Visualization)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.legend(title='Cluster', loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig('customer_segments_results/gemini3_pro/run1_poor_context/customer_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()

