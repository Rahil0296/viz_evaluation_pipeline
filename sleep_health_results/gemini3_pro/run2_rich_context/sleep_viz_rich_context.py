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

# 1. Load data
try:
    df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
except FileNotFoundError:
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# 2. Extract specified features
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

# 4. Perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

# 5. Reduce to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the visualization
plot_df = pd.DataFrame(data=X_pca, columns=['PCA 1', 'PCA 2'])
plot_df['Cluster'] = [f'Cluster {c + 1}' for c in clusters]

# 6. Create the visualization
plt.figure(figsize=(10, 7))

# Use seaborn's built-in colorblind-friendly palette
palette = sns.color_palette("colorblind", 3)

sns.scatterplot(
    x='PCA 1', 
    y='PCA 2', 
    hue='Cluster', 
    palette=palette,
    data=plot_df, 
    s=100, 
    alpha=0.8, 
    edgecolor='w'
)

# Set readable labels and title
plt.title('K-Means Clustering of Sleep Health & Lifestyle Features (k=3)', fontsize=16, pad=15)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)

# Configure legend
plt.legend(title='Clusters', title_fontsize='13', fontsize='11', loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 7. Save the plot
plt.savefig('sleep_health_results/gemini3_pro/run2_rich_context/sleep_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()