'''
Prompt:

The Ethereum transaction dataset contains behavioral and financial features of Ethereum blockchain accounts, some of which are flagged as fraudulent.
Create a visualization that identifies and highlights anomalous or fraudulent patterns in the transaction data.
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/transaction_dataset.csv")

# Select key financial features for visualization
x = df["total Ether sent"]
y = df["total ether received"]
fraud = df["FLAG"] == 1

# Create scatter plot highlighting fraudulent vs normal accounts
plt.figure()
plt.scatter(x[~fraud], y[~fraud], label="Normal Accounts", alpha=0.5)
plt.scatter(x[fraud], y[fraud], label="Fraudulent / Anomalous Accounts", alpha=0.8)

plt.xlabel("Total Ether Sent")
plt.ylabel("Total Ether Received")
plt.title("Ethereum Account Behavior: Normal vs Fraudulent Patterns")
plt.legend()

plt.savefig('defi_anomalies_results/gpt5.2/viz1_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()