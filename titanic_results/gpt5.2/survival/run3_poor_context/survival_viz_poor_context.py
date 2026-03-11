'''
Prompt: 
The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912. 
This is a sociodemographic dataset with approximately 891 passenger records. 
Create a visualization showing survival rates across different passenger classes (1st, 2nd, 3rd). 
Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/titanic.csv")  # FIX: corrected file path

# FIX: corrected column names from 'Pclass'/'Survived' to 'pclass'/'survived' (lowercase in seaborn titanic dataset)
survival_rates = df.groupby("pclass")["survived"].mean()

# Create bar chart
plt.figure()
plt.bar(survival_rates.index.astype(str), survival_rates.values)
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Titanic Survival Rates by Passenger Class")

plt.savefig('titanic_results/gpt5.2/survival/run3_poor_context/survival_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()
