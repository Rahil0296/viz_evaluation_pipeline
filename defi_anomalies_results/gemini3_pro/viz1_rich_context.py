'''
Prompt:

You are an expert data scientist and visualization specialist.

## Dataset Context:
The Ethereum Fraud Detection dataset contains aggregate transaction statistics for Ethereum blockchain accounts. Each account is labeled as fraudulent (FLAG=1) or legitimate (FLAG=0). The dataset is characterized by high sparsity (~54% zero values) and extreme outliers in Ether and ERC20 transaction values.

## Dataset Schema (use ONLY these 35 numeric features):
- Avg min between sent tnx
- Avg min between received tnx
- Time Diff between first and last (Mins)
- Sent tnx
- Received Tnx
- Number of Created Contracts
- Unique Received From Addresses
- Unique Sent To Addresses
- min value received
- max value received
- avg val received
- min val sent
- max val sent
- avg val sent
- min value sent to contract
- max val sent to contract
- avg value sent to contract
- total transactions (including tnx to create contract)
- total Ether sent
- total ether received
- total ether sent contracts
- total ether balance
- Total ERC20 tnxs
- ERC20 total Ether received
- ERC20 total ether sent
- ERC20 total Ether sent contract
- ERC20 uniq sent addr
- ERC20 uniq rec addr
- ERC20 uniq sent addr.1
- ERC20 uniq rec contract addr
- ERC20 avg time between sent tnx
- ERC20 avg time between rec tnx
- ERC20 avg time between rec 2 tnx
- ERC20 avg time between contract tnx
- ERC20 min val rec

## Technical Requirements:
- Use Python with pandas, matplotlib, or seaborn
- Load data from 'data/transaction_dataset.csv'
- Use ONLY the specified 35 features plus FLAG column
- Drop rows with missing values
- Standardize the 35 features
- Reduce to 2D using PCA
- Color-encode points by FLAG (0 = Legitimate, 1 = Fraud)
- Save the final visualization as 'output.png' with DPI=300
- Include all necessary imports
- Make the code fully executable

## Design Principles:
- Use a colorblind-friendly two-color palette (one color per class)
- Use readable axis labels and title
- Include a legend labeling Legitimate and Fraud clearly
- Title must reference Ethereum fraud detection

## Your Task: Anomaly Detection Visualization

Create a 2D PCA scatter plot that separates fraudulent from legitimate Ethereum accounts, making anomalous accounts visually distinct.

**Specific Requirements:**
- FLAG column must be used for color encoding only, not as a feature
- Fraudulent accounts must be visually distinct from legitimate ones
- Include count of each class in the legend labels

Provide ONLY the Python code wrapped in triple backticks.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Data
file_path = 'data/transaction_dataset.csv'
# Fallback in case the file is in the current directory
if not os.path.exists(file_path) and os.path.exists('transaction_dataset.csv'):
    file_path = 'transaction_dataset.csv'

df = pd.read_csv(file_path)

# Clean column names to ensure exact matching (removing trailing/leading spaces)
df.columns = [str(col).strip() for col in df.columns]

# 2. Define Features
features = [
    "Avg min between sent tnx",
    "Avg min between received tnx",
    "Time Diff between first and last (Mins)",
    "Sent tnx",
    "Received Tnx",
    "Number of Created Contracts",
    "Unique Received From Addresses",
    "Unique Sent To Addresses",
    "min value received",
    "max value received",
    "avg val received",
    "min val sent",
    "max val sent",
    "avg val sent",
    "min value sent to contract",
    "max val sent to contract",
    "avg value sent to contract",
    "total transactions (including tnx to create contract",
    "total Ether sent",
    "total ether received",
    "total ether sent contracts",
    "total ether balance",
    "Total ERC20 tnxs",
    "ERC20 total Ether received",
    "ERC20 total ether sent",
    "ERC20 total Ether sent contract",
    "ERC20 uniq sent addr",
    "ERC20 uniq rec addr",
    "ERC20 uniq sent addr.1",
    "ERC20 uniq rec contract addr",
    "ERC20 avg time between sent tnx",
    "ERC20 avg time between rec tnx",
    "ERC20 avg time between rec 2 tnx",
    "ERC20 avg time between contract tnx",
    "ERC20 min val rec"
]

# Ensure features list matches the cleaned column names (e.g. fixing missing parenthesis if needed)
features = [f.strip() for f in features]

# 3. Filter DataFrame and Drop Missing Values
cols_to_keep = features + ['FLAG']
df = df[cols_to_keep].dropna()

# 4. Standardize the 35 Features
X = df[features]
y = df['FLAG']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Reduce to 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame for plotting
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['FLAG'] = y.values

# 6. Prepare Labels and Counts
count_legit = (df_pca['FLAG'] == 0).sum()
count_fraud = (df_pca['FLAG'] == 1).sum()

df_pca['Class'] = df_pca['FLAG'].map({
    0: f'Legitimate (n={count_legit})', 
    1: f'Fraud (n={count_fraud})'
})

# 7. Visualize Data
plt.figure(figsize=(12, 8))

# Use a colorblind-friendly two-color palette (Blue for Legitimate, Vermilion for Fraud)
cb_palette = {'Legitimate': '#0072B2', 'Fraud': '#D55E00'}
palette = {
    f'Legitimate (n={count_legit})': cb_palette['Legitimate'],
    f'Fraud (n={count_fraud})': cb_palette['Fraud']
}

sns.scatterplot(
    x='PC1', 
    y='PC2', 
    hue='Class', 
    palette=palette, 
    data=df_pca, 
    alpha=0.6, 
    edgecolor='k',
    linewidth=0.3
)

# Set titles and labels
plt.title('Ethereum Fraud Detection: PCA Anomaly Visualization', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Format the legend
plt.legend(title='Account Classification', title_fontsize='12', fontsize='11', loc='best', frameon=True)

# 8. Save the final visualization
plt.tight_layout()

plt.savefig('defi_anomalies_results/gemini3_pro/viz1_rich_context_output.png', dpi=300)