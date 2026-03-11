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

# Load data
df = pd.read_csv('data/transaction_dataset.csv')

# Strip column names after loading
df.columns = df.columns.str.strip()

# Select only the required features and FLAG
features = [
    'Sent tnx', 
    'Received Tnx', 
    'Unique Sent To Addresses', 
    'Unique Received From Addresses'
]
df = df[features + ['FLAG']]

# Drop rows with missing values
df = df.dropna()

# Compute account counts for the legend
count_legit = (df['FLAG'] == 0).sum()
count_fraud = (df['FLAG'] == 1).sum()

# Compute the mean value of each feature separately for Fraud and Legitimate accounts
means = df.groupby('FLAG').mean()

# Set up the bar chart
x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for Legitimate and Fraud side by side
rects1 = ax.bar(
    x - width/2, 
    means.loc[0], 
    width, 
    label=f'Legitimate (n={count_legit})', 
    color='#0072B2'
)
rects2 = ax.bar(
    x + width/2, 
    means.loc[1], 
    width, 
    label=f'Fraud (n={count_fraud})', 
    color='#D55E00'
)

# Add text, labels, title, and custom x-axis tick labels
ax.set_ylabel('Mean Value')
ax.set_title('Ethereum Fraud Detection: Feature Anomaly Detection')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right')
ax.legend()

plt.tight_layout()

# Save the final visualization
plt.savefig('defi_anomalies_results/gemini3_pro/run3_rich_context/viz1_rich_context_output.png', dpi=300)
plt.close()