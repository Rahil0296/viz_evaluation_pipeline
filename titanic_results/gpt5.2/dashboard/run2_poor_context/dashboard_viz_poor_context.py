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

# Load dataset
df = pd.read_csv("data/titanic.csv")  

# Basic preprocessing
df['age'] = df['age'].fillna(df['age'].median())  # FIX: lowercase
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])  # FIX: lowercase

sns.set(style="whitegrid")

# Create dashboard layout
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Titanic Survival Analysis Dashboard", fontsize=18, fontweight="bold")

# 1. Survival count
sns.countplot(data=df, x="survived", ax=axes[0,0])  # FIX: lowercase
axes[0,0].set_title("Overall Survival Count")
axes[0,0].set_xticklabels(["Did Not Survive", "Survived"])

# 2. Survival by Sex
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0,1])  # FIX: lowercase
axes[0,1].set_title("Survival by Sex")

# 3. Survival by Passenger Class
sns.countplot(data=df, x="pclass", hue="survived", ax=axes[0,2])  # FIX: lowercase
axes[0,2].set_title("Survival by Passenger Class")

# 4. Age distribution by survival
sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True, ax=axes[1,0])  # FIX: lowercase
axes[1,0].set_title("Age Distribution by Survival")

# 5. Fare distribution by survival
sns.boxplot(data=df, x="survived", y="fare", ax=axes[1,1])  # FIX: lowercase
axes[1,1].set_title("Fare vs Survival")
axes[1,1].set_xticklabels(["Did Not Survive", "Survived"])

# 6. Survival by Embarkation Port
sns.countplot(data=df, x="embarked", hue="survived", ax=axes[1,2])  # FIX: lowercase
axes[1,2].set_title("Survival by Embarkation Port")

plt.tight_layout(rect=[0, 0, 1, 0.96])

# FIX: correct savefig path
plt.savefig('titanic_results/gpt5.2/dashboard/run2_poor_context/dashboard_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()