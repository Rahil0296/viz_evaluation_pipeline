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

df = pd.read_csv('data/transaction_dataset.csv')
df.columns = df.columns.str.strip()  # FIX: strip leading/trailing spaces from column names

# FIX: corrected column names from 'Sent_tnx'/'Received_tnx' to 'Sent tnx'/'Received Tnx'
avg_values = df.groupby('FLAG')[['Sent tnx', 'Received Tnx']].mean()

labels = ['Legitimate', 'Fraudulent']
sent_avg = avg_values.loc[[0, 1], 'Sent tnx']
recv_avg = avg_values.loc[[0, 1], 'Received Tnx']

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, sent_avg, width, label='Avg Sent Transactions')
plt.bar(x + width/2, recv_avg, width, label='Avg Received Transactions')
plt.xticks(x, labels)
plt.xlabel('Account Type')
plt.ylabel('Average Number of Transactions')
plt.title('Average Transactions Sent vs Received by Account Type')
plt.legend()
plt.tight_layout()

# replaced plt.show() with savefig and corrected save path
plt.savefig('defi_anomalies_results/gpt5.2/run1_poor_context/viz1_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()