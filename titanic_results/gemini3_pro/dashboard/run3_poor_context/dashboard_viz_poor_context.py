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

# Load the Titanic dataset
df = pd.read_csv('data/titanic.csv')

# Set the visualization style
sns.set_theme(style="whitegrid")

# Create a 2x2 subplot grid for the dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Titanic Survival Patterns Dashboard', fontsize=20, weight='bold', y=0.95)

# 1. Survival Rate by Passenger Class
sns.barplot(data=df, x='pclass', y='survived', ax=axes[0, 0], palette='viridis', errorbar=None)
axes[0, 0].set_title('Survival Rate by Passenger Class', fontsize=14)
axes[0, 0].set_xlabel('Passenger Class', fontsize=12)
axes[0, 0].set_ylabel('Survival Rate', fontsize=12)

# 2. Survival Rate by Sex
sns.barplot(data=df, x='sex', y='survived', ax=axes[0, 1], palette='Set2', errorbar=None)
axes[0, 1].set_title('Survival Rate by Sex', fontsize=14)
axes[0, 1].set_xlabel('Sex', fontsize=12)
axes[0, 1].set_ylabel('Survival Rate', fontsize=12)

# 3. Age Distribution by Survival Status
sns.histplot(data=df, x='age', hue='survived', multiple='stack', ax=axes[1, 0], palette='muted', bins=30)
axes[1, 0].set_title('Age Distribution by Survival', fontsize=14)
axes[1, 0].set_xlabel('Age', fontsize=12)
axes[1, 0].set_ylabel('Passenger Count', fontsize=12)
# Adjust legend for histplot
handles = axes[1, 0].get_legend().legend_handles
axes[1, 0].legend(handles=handles, title='Survived', labels=['Yes (1)', 'No (0)'])

# 4. Survival Rate by Embarkation Port and Class
sns.barplot(data=df, x='embarked', y='survived', hue='pclass', ax=axes[1, 1], palette='pastel', errorbar=None)
axes[1, 1].set_title('Survival Rate by Embarkation Port & Class', fontsize=14)
axes[1, 1].set_xlabel('Embarkation Port (C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=12)
axes[1, 1].set_ylabel('Survival Rate', fontsize=12)
axes[1, 1].legend(title='Passenger Class')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save the final visualization
plt.savefig('titanic_results/gemini3_pro/dashboard/run3_poor_context/dashboard_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()