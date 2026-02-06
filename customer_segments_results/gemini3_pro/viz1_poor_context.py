'''
The Customer Segments dataset contains information about customers and their purchasing behavior. 
Create a visualization that shows the main patterns or clusters in the data. 
Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load the dataset (FIXED PATH)
df = pd.read_csv('data/customer_segments.csv')

# 2. Data Preprocessing & Feature Engineering
# Drop missing values (mostly in Income)
df = df.dropna()

# Calculate Age (fixed reference year for reproducibility)
df['Age'] = 2015 - df['Year_Birth']

# Calculate Total Spend
df['Total_Spent'] = (
    df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
    df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
)

# Convert Dt_Customer to datetime and compute Tenure
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
last_date = df['Dt_Customer'].max()
df['Tenure'] = (last_date - df['Dt_Customer']).dt.days

# Select numerical features for clustering
features = [
    'Income', 'Recency', 'Age', 'Total_Spent', 'Tenure',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'Kidhome', 'Teenhome'
]

X = df[features]

# 3. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. PCA for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(principal_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster'].values

# 6. Visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='Cluster',
    palette='viridis',
    data=pca_df,
    s=100,
    alpha=0.7
)

plt.title('Customer Segments Visualization (PCA + K-Means)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)

# SAVE OUTPUT (FIXED NAME & PATH)
plt.tight_layout()
plt.savefig('customer_segments_results/gemini3_pro/viz1_poor_context_output.png', dpi=300)
plt.close()
