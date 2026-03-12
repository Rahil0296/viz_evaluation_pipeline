''' 
PROMPT : 

You are an expert data scientist and visualization specialist.

## Dataset Context:
The Sleep Health and Lifestyle dataset contains records describing individuals’ sleep patterns and lifestyle behaviors.

## Dataset Schema (subset used for clustering):
Use ONLY the following five numeric features:
- Sleep Duration (hours per night)
- Quality of Sleep (integer score)
- Physical Activity Level (integer score)
- Stress Level (integer score)
- Age (years)

## Technical Requirements:
- Use Python with pandas, matplotlib, seaborn, or plotly
- Load data from 'data/Sleep_health_and_lifestyle_dataset.csv'
- Use ONLY the five specified features
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
Create a 2D visualization that clusters individuals based on the specified sleep and lifestyle features.

Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Select only the specified features
features = [
    'Sleep Duration',
    'Quality of Sleep',
    'Physical Activity Level',
    'Stress Level',
    'Age'
]

X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality using PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Prepare DataFrame for plotting
plot_df = pd.DataFrame({
    'PCA Component 1': X_pca[:, 0],
    'PCA Component 2': X_pca[:, 1],
    'Cluster': clusters
})

# Colorblind-friendly palette
palette = sns.color_palette("colorblind", 3)

# Plot
plt.figure(figsize=(8, 6))
for cluster in range(3):
    subset = plot_df[plot_df['Cluster'] == cluster]
    plt.scatter(
        subset['PCA Component 1'],
        subset['PCA Component 2'],
        label=f'Cluster {cluster}',
        color=palette[cluster],
        s=60,
        alpha=0.8
    )

plt.title('Sleep & Lifestyle Clusters (PCA + K-Means)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.tight_layout()

# Save figure
plt.savefig('sleep_health_results/gpt5.2/run3_rich_context/sleep_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()