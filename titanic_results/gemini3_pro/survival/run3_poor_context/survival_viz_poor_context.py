'''
Prompt : 
The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912. 
This is a sociodemographic dataset with approximately 891 passenger records. 
Create a visualization showing survival rates across different passenger classes (1st, 2nd, 3rd). 
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('data/titanic.csv')

# Calculate the survival rate by passenger class
survival_rates = df.groupby('pclass')['survived'].mean() * 100

# Set up the figure
plt.figure(figsize=(8, 6))

# Create a bar chart
bars = plt.bar(['1st Class', '2nd Class', '3rd Class'], survival_rates, color=['#4C72B0', '#DD8452', '#55A868'])

# Add labels and title
plt.title('Survival Rates by Passenger Class on the Titanic', fontsize=14, pad=15)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate (%)', fontsize=12)
plt.ylim(0, 100)

# Add data labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1.5, 
             f'{height:.1f}%', ha='center', va='bottom', fontsize=11)

# Save the plot
plt.tight_layout()

plt.savefig('titanic_results/gemini3_pro/survival/run3_poor_context/survival_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()