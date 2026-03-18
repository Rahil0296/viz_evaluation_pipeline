'''
Prompt: 
The EV Charging Patterns dataset contains session-level information about electric vehicle charging behavior including energy usage, duration, and vehicle characteristics.
Create a visualization showing how different features relate to and impact energy consumption.
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/ev_charging_patterns.csv')

# Keep only numeric columns
numeric_df = df.select_dtypes(include='number')

# Compute correlation matrix
corr = numeric_df.corr()

# Plot heatmap
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation Matrix of EV Charging Features')

plt.savefig('ev_charging_results/gpt5.2/run3_poor_context/ev_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()