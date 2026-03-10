'''
Prompt: 

The Ethereum transaction dataset contains behavioral and financial features of Ethereum blockchain accounts, some of which are flagged as fraudulent.

Create a grouped bar chart that compares the average number of transactions sent and received between fraudulent and legitimate accounts.

Load data from 'data/transaction_dataset.csv'. Use FLAG as the label column (0 = Legitimate, 1 = Fraud).

Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('data/transaction_dataset.csv')

df.columns = df.columns.str.strip()

# Group by FLAG and compute averages
grouped = df.groupby('FLAG')[['Sent tnx', 'Received Tnx']].mean()

# Labels
labels = ['Legitimate (0)', 'Fraud (1)']
sent_avg = grouped['Sent tnx'].values
received_avg = grouped['Received Tnx'].values

# Bar positions
x = np.arange(len(labels))
width = 0.35

# Plot
plt.figure(figsize=(8,6))
plt.bar(x - width/2, sent_avg, width, label='Avg Sent Transactions')
plt.bar(x + width/2, received_avg, width, label='Avg Received Transactions')

# Formatting
plt.xticks(x, labels)
plt.ylabel('Average Number of Transactions')
plt.title('Average Sent vs Received Transactions by Account Type')
plt.legend()

plt.tight_layout()

plt.savefig('defi_anomalies_results/gpt5.2/run2_poor_context/viz1_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()