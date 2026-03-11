'''
Prompt : 
The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912. 
This is a sociodemographic dataset with approximately 891 passenger records. 
Create a visualization showing survival rates across different passenger classes (1st, 2nd, 3rd). 
Provide ONLY the Python code wrapped in triple backticks.

'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/titanic.csv')

# Calculate the survival rate for each passenger class
survival_rate = df.groupby('pclass')['survived'].mean().reset_index()

# Map the class numbers to descriptive labels
survival_rate['class_label'] = survival_rate['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

# Create the bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x='class_label', y='survived', data=survival_rate, palette='Blues_d')

# Add labels and title
plt.title('Survival Rates by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.ylim(0, 1) # Survival rate is between 0 and 1

# Display the values on top of the bars
for index, row in survival_rate.iterrows():
    plt.text(index, row.survived + 0.02, f'{row.survived:.1%}', color='black', ha='center', fontsize=11)

# Save the plot
plt.savefig('titanic_results/gemini3_pro/survival/run2_poor_context/survival_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()