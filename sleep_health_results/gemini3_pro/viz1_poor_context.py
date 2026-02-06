import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
#df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# --- Data Preprocessing ---

# Handle missing values in 'Sleep Disorder' (assuming NaN implies no disorder)
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# Split 'Blood Pressure' into Systolic and Diastolic numerical features
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

# Standardize 'BMI Category' labels (e.g., merging 'Normal Weight' into 'Normal')
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

# Drop columns not suitable or needed for clustering
df_clustering = df.drop(columns=['Person ID', 'Blood Pressure'])

# Identify categorical and numerical columns
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
numerical_cols = [col for col in df_clustering.columns if col not in categorical_cols]

# Create a column transformer for preprocessing
# - Standardize numerical features
# - One-Hot Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

# --- Clustering Model ---

# Define the clustering pipeline with K-Means
# n_clusters=3 is used as an example
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42, n_init=10))
])

# Fit the model
pipeline.fit(df_clustering)

# Get cluster labels
cluster_labels = pipeline.named_steps['cluster'].labels_

# --- Visualization ---

# Transform data for PCA visualization
X_processed = pipeline.named_steps['preprocessor'].transform(df_clustering)

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Create a DataFrame for plotting
plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plot_df['Cluster'] = cluster_labels

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Cluster', 
    data=plot_df, 
    palette='viridis', 
    s=100, 
    alpha=0.8
)
plt.title('Clustering of Individuals based on Sleep and Lifestyle Characteristics')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)

plt.savefig("sleep_health_results/gemini3_pro/viz1_poor_context_output.png", dpi=300, bbox_inches="tight")
plt.close()
