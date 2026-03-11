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
import os

# Load the dataset (handling both the requested path and the local root path)
file_path = 'data/transaction_dataset.csv'
if not os.path.exists(file_path):
    file_path = 'transaction_dataset.csv'

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

# Calculate the average number of sent and received transactions grouped by FLAG
avg_tnx = df.groupby('FLAG')[['Sent tnx', 'Received Tnx']].mean()

# Define labels and data for the bar chart
labels = ['Legitimate (0)', 'Fraud (1)']
sent_means = avg_tnx['Sent tnx']
received_means = avg_tnx['Received Tnx']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, sent_means, width, label='Sent Transactions')
rects2 = ax.bar(x + width/2, received_means, width, label='Received Transactions')

# Add text for labels, title and custom x-axis tick labels
ax.set_ylabel('Average Number of Transactions')
ax.set_title('Average Sent and Received Transactions:\nLegitimate vs Fraudulent Accounts')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display values on top of the bars
ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

plt.tight_layout()

plt.savefig('defi_anomalies_results/gemini3_pro/run2_poor_context/viz1_poor_context_output.png', dpi=300)
plt.close()