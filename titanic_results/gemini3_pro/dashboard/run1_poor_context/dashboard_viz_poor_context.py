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

# Create a more descriptive column for survival to use in legends
df['Survival Status'] = df['survived'].map({0: 'Perished', 1: 'Survived'})

# Set seaborn theme for aesthetics
sns.set_theme(style="whitegrid", palette="muted")

# Initialize the matplotlib figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Titanic Survival Patterns Dashboard', fontsize=24, fontweight='bold', y=0.98)

# Plot 1: Survival counts by Sex
sns.countplot(data=df, x='sex', hue='Survival Status', ax=axes[0, 0])
axes[0, 0].set_title('Survival by Gender', fontsize=14)
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Count')

# Plot 2: Survival counts by Passenger Class
sns.countplot(data=df, x='pclass', hue='Survival Status', ax=axes[0, 1])
axes[0, 1].set_title('Survival by Passenger Class', fontsize=14)
axes[0, 1].set_xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)')
axes[0, 1].set_ylabel('Count')

# Plot 3: Age distribution by Survival
sns.histplot(data=df, x='age', hue='Survival Status', multiple='stack', bins=30, ax=axes[0, 2])
axes[0, 2].set_title('Age Distribution by Survival', fontsize=14)
axes[0, 2].set_xlabel('Age')
axes[0, 2].set_ylabel('Count')

# Plot 4: Fare distribution by Survival
sns.boxplot(data=df, x='Survival Status', y='fare', hue='Survival Status', dodge=False, ax=axes[1, 0], legend=False)
axes[1, 0].set_ylim(-5, 150)  # Zooming in to exclude extreme outliers for better readability
axes[1, 0].set_title('Fare Distribution by Survival (Zoomed in)', fontsize=14)
axes[1, 0].set_xlabel('Survival Status')
axes[1, 0].set_ylabel('Fare')

# Plot 5: Survival counts by Embarkation Town
sns.countplot(data=df, x='embark_town', hue='Survival Status', ax=axes[1, 1])
axes[1, 1].set_title('Survival by Embarkation Town', fontsize=14)
axes[1, 1].set_xlabel('Embarkation Town')
axes[1, 1].set_ylabel('Count')

# Plot 6: Survival counts by Traveling Alone vs With Family
sns.countplot(data=df, x='alone', hue='Survival Status', ax=axes[1, 2])
axes[1, 2].set_title('Survival by Solo Travel Status', fontsize=14)
axes[1, 2].set_xlabel('Traveling Alone')
axes[1, 2].set_ylabel('Count')

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(top=0.90)

# Save the plot as requested
plt.savefig('titanic_results/gemini3_pro/dashboard/run1_poor_context/dashboard_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()