'''
Prompt: 

The Ethereum transaction dataset contains behavioral and financial features of Ethereum blockchain accounts, some of which are flagged as fraudulent.

Create a grouped bar chart that compares the average number of transactions sent and received between fraudulent and legitimate accounts.

Load data from 'data/transaction_dataset.csv'. Use FLAG as the label column (0 = Legitimate, 1 = Fraud).

Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/transaction_dataset.csv')

df.columns = df.columns.str.strip()

# Calculate the average sent and received transactions grouped by FLAG
avg_tnx = df.groupby('FLAG')[['Sent tnx', 'Received Tnx']].mean()

# Setup data for plotting
labels = ['Legitimate (0)', 'Fraud (1)']
sent_means = avg_tnx['Sent tnx'].values
received_means = avg_tnx['Received Tnx'].values

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, sent_means, width, label='Sent Transactions')
rects2 = ax.bar(x + width/2, received_means, width, label='Received Transactions')

# Add labels, title, and x-axis tick labels
ax.set_ylabel('Average Number of Transactions')
ax.set_title('Average Sent vs Received Transactions by Account Type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

plt.savefig('defi_anomalies_results/gemini3_pro/run3_poor_context/viz1_poor_context_output.png', dpi=300)
plt.close()