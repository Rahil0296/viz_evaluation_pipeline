'''
POOR CONTEXT PROMPT USED:

The EV Charging Patterns dataset contains session-level information about electric vehicle
charging behavior including energy usage, duration, and vehicle characteristics.
Create a visualization showing how different features relate to and impact energy consumption.
Provide ONLY the Python code wrapped in triple backticks.

FIXES APPLIED :
FIX 1: Removed the scatter plot loop — kept only the correlation heatmap as the single output
        (loop would generate ~10 separate unsaved figures with no defined single output)

EXECUTION VERDICT: Fail 
REASON: GPT produced a loop of scatter plots with no defined single output, making the
        visualization incoherent for the task. Moreover, GPT guessed a wrong column
        name ('energy_consumed') which silently falls back to an arbitrary last numeric
        column rather than the actual 'Energy Consumed (kWh)'. Combined, the original
        code fails to produce a valid single visualization as requested.
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/ev_charging_patterns.csv')

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Compute correlation matrix
corr = numeric_df.corr()

# FIX 1: Kept only the correlation heatmap and removed the scatter plot loop
# (loop produced multiple unsaved figures with no defined single output)
plt.figure()
plt.imshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Matrix of EV Charging Features")
plt.tight_layout()

plt.savefig('ev_charging_results/gpt5.2/run2_poor_context/ev_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()