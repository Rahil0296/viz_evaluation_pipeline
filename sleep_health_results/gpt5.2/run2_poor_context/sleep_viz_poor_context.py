''' 
PROMPT : 

The Sleep Health and Lifestyle dataset contains information about individuals’ sleep patterns and lifestyle factors. 
Create a visualization that clusters individuals based on their sleep and lifestyle characteristics. 
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop ID column if present
if "Person ID" in df.columns:
    df = df.drop(columns=["Person ID"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustering of Individuals Based on Sleep and Lifestyle Characteristics")

plt.savefig('sleep_health_results/gpt5.2/run2_poor_context/sleep_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()