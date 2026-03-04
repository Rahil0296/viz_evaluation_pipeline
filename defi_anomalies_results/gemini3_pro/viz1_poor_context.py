'''
Prompt:

The Ethereum transaction dataset contains behavioral and financial features of Ethereum blockchain accounts, some of which are flagged as fraudulent.
Create a visualization that identifies and highlights anomalous or fraudulent patterns in the transaction data.
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/transaction_dataset.csv')

# Drop rows where critical financial features or FLAG might be missing
features_to_plot = ['total ether received', 'total Ether sent', 'FLAG']
df_plot = df[features_to_plot].dropna()

# Apply log1p transformation to handle highly skewed transaction values
df_plot['Log Total Ether Received'] = np.log1p(df_plot['total ether received'])
df_plot['Log Total Ether Sent'] = np.log1p(df_plot['total Ether sent'])

# Set up the plot style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

# Create scatter plot highlighting anomalous/fraudulent patterns (FLAG=1)
scatter = sns.scatterplot(
    data=df_plot,
    x='Log Total Ether Received',
    y='Log Total Ether Sent',
    hue='FLAG',
    palette={0: '#1f77b4', 1: '#d62728'}, # Blue for normal, Red for fraud
    alpha=0.6,
    s=30,
    edgecolor=None
)

# Customize labels and title
plt.title('Anomaly Detection: Total Ether Sent vs. Received (Log Scale)', fontsize=16, pad=15)
plt.xlabel('Log(Total Ether Received + 1)', fontsize=12)
plt.ylabel('Log(Total Ether Sent + 1)', fontsize=12)

# Update legend
handles, _ = scatter.get_legend_handles_labels()
plt.legend(handles=handles, labels=['Normal (0)', 'Fraudulent (1)'], title='Account Type', loc='upper right')

plt.tight_layout()

# Save the visualization
plt.savefig('defi_anomalies_results/gemini3_pro/viz1_poor_context_output.png', dpi=300)
