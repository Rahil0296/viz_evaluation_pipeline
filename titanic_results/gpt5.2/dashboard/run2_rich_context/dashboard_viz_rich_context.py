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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/titanic.csv")

# -----------------------------
# Handle Missing Values
# -----------------------------
# Fill age with median
df["age"] = df["age"].fillna(df["age"].median())

# Fill embarked with mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# -----------------------------
# Feature Engineering
# -----------------------------
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Map survival for readability
df["survived_label"] = df["survived"].map({0: "Did Not Survive", 1: "Survived"})

# -----------------------------
# Styling
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")

# -----------------------------
# Create Dashboard (2x3)
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# -----------------------------
# 1. Survival by Passenger Class
# -----------------------------
sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    ax=axes[0, 0],
    estimator=np.mean,
    palette="Blues_d"
)
axes[0, 0].set_title("Survival Rate by Passenger Class")
axes[0, 0].set_xlabel("Passenger Class")
axes[0, 0].set_ylabel("Survival Rate")

# -----------------------------
# 2. Survival by Gender
# -----------------------------
sns.barplot(
    data=df,
    x="sex",
    y="survived",
    ax=axes[0, 1],
    estimator=np.mean,
    palette="Set2"
)
axes[0, 1].set_title("Survival Rate by Gender")
axes[0, 1].set_xlabel("Gender")
axes[0, 1].set_ylabel("Survival Rate")

# -----------------------------
# 3. Age Distribution
# -----------------------------
sns.histplot(
    df["age"],
    bins=30,
    kde=True,
    ax=axes[0, 2],
    color="steelblue"
)
axes[0, 2].set_title("Age Distribution of Passengers")
axes[0, 2].set_xlabel("Age")
axes[0, 2].set_ylabel("Count")

# -----------------------------
# 4. Fare Distribution
# -----------------------------
sns.histplot(
    df["fare"],
    bins=40,
    kde=True,
    ax=axes[1, 0],
    color="darkorange"
)
axes[1, 0].set_title("Fare Distribution")
axes[1, 0].set_xlabel("Fare")
axes[1, 0].set_ylabel("Count")

# -----------------------------
# 5. Survival by Family Size
# -----------------------------
family_survival = df.groupby("family_size")["survived"].mean().reset_index()

sns.barplot(
    data=family_survival,
    x="family_size",
    y="survived",
    ax=axes[1, 1],
    palette="viridis"
)
axes[1, 1].set_title("Survival Rate by Family Size")
axes[1, 1].set_xlabel("Family Size")
axes[1, 1].set_ylabel("Survival Rate")

# -----------------------------
# 6. Survival by Embarkation Port
# -----------------------------
sns.barplot(
    data=df,
    x="embarked",
    y="survived",
    ax=axes[1, 2],
    estimator=np.mean,
    palette="magma"
)
axes[1, 2].set_title("Survival Rate by Embarkation Port")
axes[1, 2].set_xlabel("Embarkation Port")
axes[1, 2].set_ylabel("Survival Rate")

# -----------------------------
# Main Title and Layout
# -----------------------------
fig.suptitle(
    "Titanic Disaster: Comprehensive Survival Analysis",
    fontsize=24,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# -----------------------------
# Save Figure
# -----------------------------
plt.savefig('titanic_results/gpt5.2/dashboard/run2_rich_context/dashboard_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()