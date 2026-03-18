'''
Prompt:
The EV Charging Patterns dataset contains session-level information about electric vehicle charging behavior including energy usage, duration, and vehicle characteristics.
Create a visualization showing how different features relate to and impact energy consumption.
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/ev_charging_patterns.csv')

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Create a heatmap to visualize how features relate to Energy Consumed and each other
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap of EV Charging Features', fontsize=16)
plt.tight_layout()

# Save the plot
plt.savefig('ev_charging_results/gemini3_pro/viz1_poor_context_output.png', dpi=300, bbox_inches='tight')
