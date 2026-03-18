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

# Select numerical columns to analyze relationships
numeric_cols = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numeric_cols.corr()

# Isolate correlations specifically with 'Energy Consumed (kWh)' and sort them
target_corr = corr_matrix[['Energy Consumed (kWh)']].drop('Energy Consumed (kWh)').sort_values(by='Energy Consumed (kWh)', ascending=False)

# Create a figure with two subplots: full heatmap and target bar chart
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Subplot 1: Correlation Heatmap of all numeric features
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title('Correlation Heatmap of EV Charging Features', fontsize=14)

# Subplot 2: Bar plot focusing on 'Energy Consumed'
sns.barplot(x=target_corr['Energy Consumed (kWh)'], y=target_corr.index, ax=axes[1], hue=target_corr.index, legend=False, palette='coolwarm')
axes[1].set_title('Impact/Correlation with Energy Consumed (kWh)', fontsize=14)
axes[1].set_xlabel('Pearson Correlation Coefficient', fontsize=12)
axes[1].set_ylabel('Features', fontsize=12)
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('ev_charging_results/gemini3_pro/run3_poor_context/ev_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()