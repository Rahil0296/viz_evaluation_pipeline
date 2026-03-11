'''
Prompt:

The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912. 
Create a comprehensive dashboard visualization showing multiple different insights about 
survival patterns in the data. Use matplotlib or seaborn only. Save the final visualization 
as 'output.png'. Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/titanic.csv')

# Set seaborn style
sns.set_theme(style="whitegrid")

# Create a figure and a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Titanic Survival Patterns Dashboard', fontsize=20, fontweight='bold', y=0.98)

# 1. Survival count by Sex
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0, 0], palette='pastel')
axes[0, 0].set_title('Survival Count by Sex', fontsize=14)
axes[0, 0].set_xlabel('Sex', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].legend(title='Survived', labels=['No', 'Yes'])

# 2. Survival count by Passenger Class
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title('Survival Count by Passenger Class', fontsize=14)
axes[0, 1].set_xlabel('Passenger Class', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)
axes[0, 1].legend(title='Survived', labels=['No', 'Yes'])

# 3. Age Distribution by Survival
sns.histplot(data=df, x='age', hue='survived', multiple='stack', ax=axes[1, 0], palette='pastel', bins=30, kde=True)
axes[1, 0].set_title('Age Distribution by Survival', fontsize=14)
axes[1, 0].set_xlabel('Age', fontsize=12)
axes[1, 0].set_ylabel('Count', fontsize=12)
axes[1, 0].legend(title='Survived', labels=['Yes', 'No'])

# 4. Fare Distribution by Survival (Outliers hidden for better visualization)
sns.boxplot(data=df, x='survived', y='fare', hue='survived', ax=axes[1, 1], palette='pastel', showfliers=False, legend=False)
axes[1, 1].set_title('Fare Distribution by Survival', fontsize=14)
axes[1, 1].set_xlabel('Survived', fontsize=12)
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['No', 'Yes'])
axes[1, 1].set_ylabel('Fare', fontsize=12)

# Adjust layout and save the plot
plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.savefig('titanic_results/gemini3_pro/dashboard/run2_poor_context/dashboard_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()