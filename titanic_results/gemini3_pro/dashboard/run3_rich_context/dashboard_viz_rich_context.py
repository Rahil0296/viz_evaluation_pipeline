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

# Set visualization style
sns.set_theme(style="whitegrid")

# Load data (handles both specified and actual file paths)
file_path = 'data/titanic.csv' if os.path.exists('data/titanic.csv') else 'titanic.csv'
df = pd.read_csv(file_path)

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['fare'] = df['fare'].fillna(df['fare'].median())

# Create family size feature
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Create figure and axes for a 2x3 dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Titanic Disaster: Comprehensive Survival Analysis", fontsize=22, fontweight='bold', y=0.98)

# 1. Survival by Passenger Class
sns.barplot(data=df, x='pclass', y='survived', hue='pclass', ax=axes[0, 0], palette='viridis', errorbar=None, legend=False)
axes[0, 0].set_title('Survival Rate by Passenger Class', fontsize=14)
axes[0, 0].set_xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)', fontsize=12)
axes[0, 0].set_ylabel('Survival Rate', fontsize=12)

# 2. Survival by Gender
sns.barplot(data=df, x='sex', y='survived', hue='sex', ax=axes[0, 1], palette='Set2', errorbar=None, legend=False)
axes[0, 1].set_title('Survival Rate by Gender', fontsize=14)
axes[0, 1].set_xlabel('Gender', fontsize=12)
axes[0, 1].set_ylabel('Survival Rate', fontsize=12)

# 3. Age Distribution by Survival
sns.histplot(data=df, x='age', hue='survived', multiple='stack', ax=axes[0, 2], palette=['#e74c3c', '#2ecc71'], bins=30)
axes[0, 2].set_title('Age Distribution by Survival', fontsize=14)
axes[0, 2].set_xlabel('Age (Years)', fontsize=12)
axes[0, 2].set_ylabel('Passenger Count', fontsize=12)
legend = axes[0, 2].get_legend()
if legend:
    legend.set_title('Survived')

# 4. Fare Distribution by Survival (using log scale due to skewness)
df['fare_adjusted'] = df['fare'] + 1 # Add 1 to handle 0 fares in log scale
sns.histplot(data=df, x='fare_adjusted', hue='survived', multiple='stack', ax=axes[1, 0], palette=['#e74c3c', '#2ecc71'], bins=30, log_scale=True)
axes[1, 0].set_title('Fare Distribution by Survival (Log Scale)', fontsize=14)
axes[1, 0].set_xlabel('Fare (+1)', fontsize=12)
axes[1, 0].set_ylabel('Passenger Count', fontsize=12)

# 5. Survival Rate by Family Size
sns.barplot(data=df, x='family_size', y='survived', hue='family_size', ax=axes[1, 1], palette='coolwarm', errorbar=None, legend=False)
axes[1, 1].set_title('Survival Rate by Family Size', fontsize=14)
axes[1, 1].set_xlabel('Total Family Size Aboard', fontsize=12)
axes[1, 1].set_ylabel('Survival Rate', fontsize=12)

# 6. Survival Rate by Embarkation Port
sns.barplot(data=df, x='embarked', y='survived', hue='embarked', ax=axes[1, 2], palette='pastel', errorbar=None, legend=False)
axes[1, 2].set_title('Survival Rate by Embarkation Port', fontsize=14)
axes[1, 2].set_xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)', fontsize=12)
axes[1, 2].set_ylabel('Survival Rate', fontsize=12)

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the final publication-ready visualization
plt.savefig('titanic_results/gemini3_pro/dashboard/run3_rich_context/dashboard_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()