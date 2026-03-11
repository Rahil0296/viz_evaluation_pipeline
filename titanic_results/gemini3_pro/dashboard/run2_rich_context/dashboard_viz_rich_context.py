'''
Prompt : 
You are an expert data scientist and visualization specialist.

## Dataset Context:
The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912.
This is a sociodemographic dataset with approximately 891 passenger records.

## Dataset Schema (IMPORTANT: All column names are lowercase):
- survived: Survival status (0 = No, 1 = Yes)
- pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- sex: Gender (male/female)
- age: Age in years (float, some missing values ~20%)
- sibsp: Number of siblings/spouses aboard
- parch: Number of parents/children aboard
- fare: Passenger fare (float)
- embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Technical Requirements:
- Use Python with matplotlib and seaborn only (no Plotly, no Dash)
- Load data from 'data/titanic.csv'
- Handle missing values appropriately
- Save the final visualization as 'output.png' with DPI=300
- Include all necessary imports
- Make the code fully executable

## Your Task: Comprehensive Titanic Dashboard
Create a multi-panel dashboard showing multiple different aspects of the Titanic data.

## Specific Requirements:
- Use subplots to create a 2x3 or 3x2 dashboard layout
- Include: survival by class, by gender, age distribution, fare distribution, family size, embarkation
- Make it publication-ready with consistent styling
- Add a main title: "Titanic Disaster: Comprehensive Survival Analysis"
- Ensure all subplots are properly labeled

Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data, using a fallback path just in case the directory structure differs
file_path = 'data/titanic.csv' if os.path.exists('data/titanic.csv') else 'titanic.csv'
df = pd.read_csv(file_path)

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['fare'] = df['fare'].fillna(df['fare'].median())

# Create a family size feature (siblings/spouses + parents/children)
df['family_size'] = df['sibsp'] + df['parch']

# Set publication-ready style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = "muted"

# Create the 2x3 dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Titanic Disaster: Comprehensive Survival Analysis", fontsize=22, fontweight='bold', y=0.98)

# 1. Survival by Class (Survival Rate)
sns.barplot(
    x='pclass', y='survived', data=df, ax=axes[0, 0], 
    hue='pclass', palette=palette, legend=False, capsize=.1
)
axes[0, 0].set_title("Survival Rate by Passenger Class", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Passenger Class (1=1st, 2=2nd, 3=3rd)")
axes[0, 0].set_ylabel("Survival Rate")

# 2. Survival by Gender (Survival Rate)
sns.barplot(
    x='sex', y='survived', data=df, ax=axes[0, 1], 
    hue='sex', palette=palette, legend=False, capsize=.1
)
axes[0, 1].set_title("Survival Rate by Gender", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Gender")
axes[0, 1].set_ylabel("Survival Rate")

# 3. Age Distribution by Survival
sns.histplot(
    data=df, x='age', hue='survived', multiple='stack', 
    bins=30, ax=axes[0, 2], palette=['#e74c3c', '#2ecc71']
)
axes[0, 2].set_title("Age Distribution by Survival", fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel("Age (years)")
axes[0, 2].set_ylabel("Passenger Count")
axes[0, 2].legend(title='Survived', labels=['Yes (1)', 'No (0)'])

# 4. Fare Distribution by Survival
sns.histplot(
    data=df, x='fare', hue='survived', multiple='stack', 
    bins=30, ax=axes[1, 0], palette=['#e74c3c', '#2ecc71']
)
axes[1, 0].set_title("Fare Distribution by Survival", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Fare")
axes[1, 0].set_ylabel("Passenger Count")
axes[1, 0].set_xlim(0, 200) # Limit x-axis to 200 to mitigate visual skew from extreme outliers
axes[1, 0].legend(title='Survived', labels=['Yes (1)', 'No (0)'])

# 5. Survival Rate by Family Size
sns.barplot(
    x='family_size', y='survived', data=df, ax=axes[1, 1], 
    hue='family_size', palette=palette, legend=False, errorbar=None
)
axes[1, 1].set_title("Survival Rate by Family Size", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Number of Family Members Aboard (SibSp + Parch)")
axes[1, 1].set_ylabel("Survival Rate")

# 6. Survival by Embarkation Port
sns.barplot(
    x='embarked', y='survived', data=df, ax=axes[1, 2], 
    hue='embarked', palette=palette, legend=False, capsize=.1
)
axes[1, 2].set_title("Survival Rate by Embarkation Port", fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel("Port (C=Cherbourg, Q=Queenstown, S=Southampton)")
axes[1, 2].set_ylabel("Survival Rate")

# Adjust layout to prevent overlap and ensure clean visual presentation
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the multi-panel dashboard directly to file
plt.savefig('titanic_results/gemini3_pro/dashboard/run2_rich_context/dashboard_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()