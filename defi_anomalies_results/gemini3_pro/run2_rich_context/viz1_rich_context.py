'''
Prompt: 

You are an expert data scientist and visualization specialist.

## Dataset Context:
The Ethereum Fraud Detection dataset contains aggregate transaction statistics for Ethereum blockchain accounts. Each account is labeled as fraudulent (FLAG=1) or legitimate (FLAG=0).

## Dataset Schema (use ONLY these features plus FLAG):
- Sent tnx
- Received Tnx
- Unique Sent To Addresses
- Unique Received From Addresses
- FLAG: 0 = Legitimate, 1 = Fraud

## Technical Requirements:
- Use Python with pandas, matplotlib, or seaborn
- Load data from 'data/transaction_dataset.csv'
- Strip column names after loading: df.columns = df.columns.str.strip()
- Drop rows with missing values
- Compute the mean value of each feature separately for Fraud and Legitimate accounts
- Visualize as a grouped bar chart with Fraud and Legitimate side by side for each feature
- Save the final visualization as 'output.png' with DPI=300
- Include all necessary imports
- Make the code fully executable

## Design Principles:
- Use a colorblind-friendly two-color palette (#0072B2 for Legitimate, #D55E00 for Fraud)
- Rotate x-axis labels for readability
- Include a legend labeling Legitimate and Fraud clearly with account counts in parentheses
- Title must reference Ethereum fraud detection and anomaly detection

Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (handles both the requested path and the current directory as a fallback)
try:
    df = pd.read_csv('data/transaction_dataset.csv')
except FileNotFoundError:
    df = pd.read_csv('transaction_dataset.csv')

# Strip column names after loading
df.columns = df.columns.str.strip()

# Define the features to keep based on the dataset schema provided
features = [
    'Sent tnx',
    'Received Tnx',
    'Unique Sent To Addresses',
    'Unique Received From Addresses'
]

# Keep only necessary features + FLAG, and drop missing values
df = df[features + ['FLAG']].dropna()

# Compute account counts for the legend
count_legit = (df['FLAG'] == 0).sum()
count_fraud = (df['FLAG'] == 1).sum()

# Compute the mean value of each feature grouped by FLAG
mean_df = df.groupby('FLAG').mean()

# Setup the plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(features))
width = 0.35

# Extract means for Legitimate (0) and Fraud (1) accounts
legit_means = mean_df.loc[0].values
fraud_means = mean_df.loc[1].values

# Create grouped bar charts with colorblind-friendly colors
ax.bar(x - width/2, legit_means, width, label=f'Legitimate ({count_legit})', color='#0072B2')
ax.bar(x + width/2, fraud_means, width, label=f'Fraud ({count_fraud})', color='#D55E00')

# Format the chart
ax.set_ylabel('Mean Value')
ax.set_title('Ethereum Fraud Detection: Anomaly Detection via Account Statistics')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.legend()

# Layout adjustment and save
plt.tight_layout()

plt.savefig('defi_anomalies_results/gemini3_pro/run2_rich_context/viz1_rich_context_output.png', dpi=300)
plt.close()