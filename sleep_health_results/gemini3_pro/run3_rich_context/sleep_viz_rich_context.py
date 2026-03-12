''' 
Prompt :

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

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load data
file_path = 'data/Sleep_health_and_lifestyle_dataset.csv'
# Fallback to current directory if 'data/' folder doesn't exist
if not os.path.exists(file_path):
    file_path = 'Sleep_health_and_lifestyle_dataset.csv'

df = pd.read_csv(file_path)

# 2. Use ONLY the specified features
features = [
    'Sleep Duration', 
    'Quality of Sleep', 
    'Physical Activity Level', 
    'Stress Level', 
    'Age'
]
X = df[features]

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reduce to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 5. Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# Clustering on the scaled data (could also be done on X_pca)
clusters = kmeans.fit_predict(X_scaled) 

# Prepare dataframe for plotting
plot_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
plot_df['Cluster'] = [f'Cluster {c}' for c in clusters]

# 6. Visualization setup
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# Use a colorblind-friendly palette
palette = sns.color_palette("colorblind", 3)

# Create scatter plot
ax = sns.scatterplot(
    data=plot_df,
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Cluster',
    palette=palette,
    s=100,
    alpha=0.8,
    edgecolor='w',
    linewidth=0.5
)

# Design enhancements: Titles, labels, legend
plt.title('Sleep Health & Lifestyle Clusters (PCA Projection)', fontsize=16, pad=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Legend adjustments
plt.legend(
    title='K-Means Clusters',
    title_fontsize=12,
    fontsize=11,
    loc='best',
    frameon=True,
    shadow=True
)

plt.tight_layout()

# 7. Save the final visualization
plt.savefig('sleep_health_results/gemini3_pro/run3_rich_context/sleep_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()