''' 
PROMPT : 

The Sleep Health and Lifestyle dataset contains information about individuals’ sleep patterns and lifestyle factors. 
Create a visualization that clusters individuals based on their sleep and lifestyle characteristics. 
Provide ONLY the Python code wrapped in triple backticks.

'''


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
#file_path = "/mnt/data/Sleep_health_and_lifestyle_dataset.csv"

# file path :
file_path = "data/Sleep_health_and_lifestyle_dataset.csv"


df = pd.read_csv(file_path)

# Select numerical features for clustering
features = df.select_dtypes(include=["int64", "float64"]).dropna()

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Plot clusters
plt.figure()
plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustering of Individuals Based on Sleep and Lifestyle Characteristics")

plt.tight_layout()
plt.savefig("sleep_health_results/gpt5.2/viz1_poor_context_output.png", dpi=300)
plt.close()

