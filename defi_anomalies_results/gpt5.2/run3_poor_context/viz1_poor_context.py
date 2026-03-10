'''
Prompt:


'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('data/transaction_dataset.csv')

df.columns = df.columns.str.strip()

# Columns representing number of transactions sent and received
sent_col = 'Sent tnx'
received_col = 'Received Tnx'

# Compute averages grouped by FLAG (0 = Legitimate, 1 = Fraud)
grouped = df.groupby('FLAG')[[sent_col, received_col]].mean()

# Labels
labels = ['Legitimate', 'Fraud']
x = np.arange(len(labels))
width = 0.35

# Create grouped bar chart
fig, ax = plt.subplots()

ax.bar(x - width/2, grouped[sent_col], width, label='Avg Sent Transactions')
ax.bar(x + width/2, grouped[received_col], width, label='Avg Received Transactions')

# Formatting
ax.set_ylabel('Average Number of Transactions')
ax.set_title('Average Sent vs Received Transactions by Account Type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()

plt.savefig('defi_anomalies_results/gpt5.2/run3_poor_context/viz1_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()