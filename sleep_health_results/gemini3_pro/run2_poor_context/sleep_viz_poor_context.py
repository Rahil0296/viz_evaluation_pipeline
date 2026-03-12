''' 
PROMPT : 

The Sleep Health and Lifestyle dataset contains information about individuals’ sleep patterns and lifestyle factors. 
Create a visualization that clusters individuals based on their sleep and lifestyle characteristics. 
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# 2. Select numerical features relevant to sleep and lifestyle
features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
            'Stress Level', 'Heart Rate', 'Daily Steps']

# Drop missing values if any exist in these columns
X = df[features].dropna()

# 3. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means clustering (using 3 clusters as a default)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# 5. Reduce dimensions to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Create a DataFrame for the visualization
plot_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
plot_df['Cluster'] = cluster_labels

# 7. Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Principal Component 1', 
    y='Principal Component 2', 
    hue='Cluster', 
    palette='viridis', 
    data=plot_df, 
    s=80, 
    alpha=0.8
)

plt.title('Clusters of Individuals Based on Sleep & Lifestyle Characteristics')
plt.xlabel('Principal Component 1 (Captures max variance in features)')
plt.ylabel('Principal Component 2 (Captures second max variance)')
plt.legend(title='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the visualization to a file
plt.savefig('sleep_health_results/gemini3_pro/run2_poor_context/sleep_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()