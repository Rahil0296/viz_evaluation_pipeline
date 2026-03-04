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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('data/transaction_dataset.csv')

# FIX 1: strip leading/trailing spaces from all column names
# (CSV has leading spaces on ERC20 columns and trailing space on 'max value received ')
df.columns = df.columns.str.strip()

# FIX 2: rename malformed column — missing closing parenthesis in CSV
df = df.rename(columns={
    'total transactions (including tnx to create contract':
    'total transactions (including tnx to create contract)'
})

# Define the 35 numeric features
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
    "total transactions (including tnx to create contract)",
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

# Keep only required columns
df = df[features + ["FLAG"]]

# Drop missing values
df = df.dropna()

# Separate features and labels
X = df[features]
y = df["FLAG"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convert to DataFrame for plotting
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["FLAG"] = y.values

# Class counts
counts = pca_df["FLAG"].value_counts()

# Colorblind-friendly palette
color_legit = "#0072B2"
color_fraud = "#D55E00"

# Plot
plt.figure(figsize=(10, 7))

legit = pca_df[pca_df["FLAG"] == 0]
fraud = pca_df[pca_df["FLAG"] == 1]

plt.scatter(
    legit["PC1"],
    legit["PC2"],
    c=color_legit,
    s=20,
    alpha=0.7,
    label=f"Legitimate (n={counts.get(0,0)})"
)

plt.scatter(
    fraud["PC1"],
    fraud["PC2"],
    c=color_fraud,
    s=20,
    alpha=0.9,
    label=f"Fraud (n={counts.get(1,0)})"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Ethereum Fraud Detection: PCA Visualization of Transaction Anomalies")
plt.legend()
plt.tight_layout()

# FIX 3: correct save path and remove plt.show()
plt.savefig("defi_anomalies_results/gpt5.2/viz1_rich_context_output.png", dpi=300)
plt.close()