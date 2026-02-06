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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================================
# 1. Load data
# ================================
# FIX 1 (Path alignment):
# Original Gemini code used 'Sleep_health_and_lifestyle_dataset.csv' in the working directory.
# We align this with the repository structure: data/...
file_path = 'data/Sleep_health_and_lifestyle_dataset.csv'
df = pd.read_csv(file_path)

# ================================
# 2. Select ONLY the specified features (per rich-context prompt)
# ================================
features = [
    'Sleep Duration',
    'Quality of Sleep',
    'Physical Activity Level',
    'Stress Level',
    'Age'
]
X = df[features]

# ================================
# 3. Standardize the features
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 4. Reduce to 2D using PCA
# ================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for visualization
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# ================================
# 5. Perform k-means clustering with k=3
# ================================
# FIX 2 (Protocol compliance):
# Original Gemini code clustered on X_scaled (5D space).
# This violates the prompt instruction:
#   "Reduce to 2D using PCA" → "Perform k-means clustering"
# To enforce methodological consistency across models,
# clustering is now performed on the PCA-reduced 2D representation.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

pca_df['Cluster'] = clusters

# ================================
# 6. Visualization
# ================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 7))

# Use colorblind-friendly palette
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='colorblind',
    edgecolor='w',
    s=100,
    alpha=0.8
)

plt.title('2D PCA Visualization of Sleep and Lifestyle Clusters (k=3)', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# ================================
# 7. Save the final visualization
# ================================
# FIX 3 (Reproducibility):
# Explicitly saving output as required by evaluation pipeline.
output_path = "sleep_health_results/gemini3_pro/viz1_rich_context_output.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
