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
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('data/transaction_dataset.csv')

# Strip column names
df.columns = df.columns.str.strip()

# Drop rows with missing values
df = df.dropna()

# Features to analyze
features = [
    'Sent tnx',
    'Received Tnx',
    'Unique Sent To Addresses',
    'Unique Received From Addresses'
]

# Compute mean values by FLAG
means = df.groupby('FLAG')[features].mean()

# Count accounts for legend
counts = df['FLAG'].value_counts()
legit_count = counts.get(0, 0)
fraud_count = counts.get(1, 0)

# Extract values
legit_means = means.loc[0]
fraud_means = means.loc[1]

# Plot setup
x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width/2,
    legit_means,
    width,
    color="#0072B2",
    label=f"Legitimate (n={legit_count})"
)

plt.bar(
    x + width/2,
    fraud_means,
    width,
    color="#D55E00",
    label=f"Fraud (n={fraud_count})"
)

# Labels and formatting
plt.xticks(x, features, rotation=30, ha='right')
plt.ylabel('Mean Value')
plt.title('Ethereum Fraud Detection: Transaction Behavior Anomaly Detection')
plt.legend()

plt.tight_layout()

# Save figure
plt.savefig('defi_anomalies_results/gpt5.2/run1_rich_context/viz1_rich_context_output.png', dpi=300)
plt.close()