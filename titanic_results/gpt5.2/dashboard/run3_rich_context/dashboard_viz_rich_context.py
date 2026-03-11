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

# Titanic Comprehensive Survival Analysis Dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv('data/titanic.csv')

# -----------------------------
# Data Cleaning / Preparation
# -----------------------------

# Handle missing age values by filling with median age
df['age'] = df['age'].fillna(df['age'].median())

# Handle missing embarked values by filling with most common port
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Create family size feature
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Convert survival to label for clarity in plots
df['survived_label'] = df['survived'].map({0: 'Did Not Survive', 1: 'Survived'})

# -----------------------------
# Styling
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")
palette = "Set2"

# -----------------------------
# Create Dashboard Layout
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Survival by Passenger Class
sns.barplot(
    data=df,
    x='pclass',
    y='survived',
    ax=axes[0, 0],
    palette=palette
)
axes[0, 0].set_title('Survival Rate by Passenger Class')
axes[0, 0].set_xlabel('Passenger Class')
axes[0, 0].set_ylabel('Survival Rate')

# 2. Survival by Gender
sns.barplot(
    data=df,
    x='sex',
    y='survived',
    ax=axes[0, 1],
    palette=palette
)
axes[0, 1].set_title('Survival Rate by Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Survival Rate')

# 3. Age Distribution by Survival
sns.histplot(
    data=df,
    x='age',
    hue='survived_label',
    bins=30,
    kde=True,
    ax=axes[0, 2],
    palette=palette
)
axes[0, 2].set_title('Age Distribution by Survival')
axes[0, 2].set_xlabel('Age')
axes[0, 2].set_ylabel('Count')

# 4. Fare Distribution
sns.histplot(
    df['fare'],
    bins=30,
    kde=True,
    ax=axes[1, 0],
    color='steelblue'
)
axes[1, 0].set_title('Fare Distribution')
axes[1, 0].set_xlabel('Fare')
axes[1, 0].set_ylabel('Count')

# 5. Survival by Family Size
sns.barplot(
    data=df,
    x='family_size',
    y='survived',
    ax=axes[1, 1],
    palette=palette
)
axes[1, 1].set_title('Survival Rate by Family Size')
axes[1, 1].set_xlabel('Family Size')
axes[1, 1].set_ylabel('Survival Rate')

# 6. Passenger Distribution by Embarkation Port
sns.countplot(
    data=df,
    x='embarked',
    ax=axes[1, 2],
    palette=palette
)
axes[1, 2].set_title('Passengers by Embarkation Port')
axes[1, 2].set_xlabel('Embarkation Port')
axes[1, 2].set_ylabel('Number of Passengers')

# -----------------------------
# Main Title and Layout
# -----------------------------
fig.suptitle(
    "Titanic Disaster: Comprehensive Survival Analysis",
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# -----------------------------
# Save Figure
# -----------------------------
plt.savefig('titanic_results/gpt5.2/dashboard/run3_rich_context/dashboard_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()