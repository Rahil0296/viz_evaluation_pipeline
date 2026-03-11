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

# Load data (handling both requested and actual paths)
try:
    df = pd.read_csv('data/transaction_dataset.csv')
except FileNotFoundError:
    df = pd.read_csv('transaction_dataset.csv')

df.columns = df.columns.str.strip()

# Calculate the mean of sent and received transactions grouped by FLAG
avg_data = df.groupby('FLAG')[['Sent tnx', 'Received Tnx']].mean()

# Set up the grouped bar chart
labels = ['Legitimate (FLAG=0)', 'Fraud (FLAG=1)']
sent_means = avg_data['Sent tnx'].values
received_means = avg_data['Received Tnx'].values

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, sent_means, width, label='Sent Transactions', color='skyblue')
rects2 = ax.bar(x + width/2, received_means, width, label='Received Transactions', color='salmon')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Number of Transactions')
ax.set_title('Average Sent vs Received Transactions by Account Type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on top of bars
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')

plt.tight_layout()

plt.savefig('defi_anomalies_results/gemini3_pro/run1_poor_context/viz1_poor_context_output.png', dpi=300)
plt.close()