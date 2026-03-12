''' 
PROMPT : 

The Sleep Health and Lifestyle dataset contains information about individuals’ sleep patterns and lifestyle factors. 
Create a visualization that clusters individuals based on their sleep and lifestyle characteristics. 
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Split Blood Pressure into systolic and diastolic
if "Blood Pressure" in df.columns:
    bp_split = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")
    df = df.drop(columns=["Blood Pressure"])

# Drop identifier if present
if "Person ID" in df.columns:
    df = df.drop(columns=["Person ID"])

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Handle missing values
df_encoded = df_encoded.fillna(df_encoded.mean())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustering of Individuals Based on Sleep and Lifestyle Characteristics")

plt.savefig('sleep_health_results/gpt5.2/run3_poor_context/sleep_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()