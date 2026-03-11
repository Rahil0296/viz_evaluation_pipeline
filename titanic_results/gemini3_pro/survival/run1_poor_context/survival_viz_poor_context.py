'''
Prompt : 
The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912. 
This is a sociodemographic dataset with approximately 891 passenger records. 
Create a visualization showing survival rates across different passenger classes (1st, 2nd, 3rd). 
Provide ONLY the Python code wrapped in triple backticks.

'''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/titanic.csv')

# Create a bar plot showing survival rates by passenger class
plt.figure(figsize=(8, 6))
sns.barplot(x='pclass', y='survived', data=df, palette='viridis', hue='pclass', legend=False)

# Set chart title and labels
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)

# Display the plot
plt.tight_layout()
plt.savefig('titanic_results/gemini3_pro/survival/run1_poor_context/survival_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()
