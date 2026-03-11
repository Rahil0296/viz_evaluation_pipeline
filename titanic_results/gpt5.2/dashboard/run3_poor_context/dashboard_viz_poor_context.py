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

# Set style
sns.set_theme(style="whitegrid")

# Create dashboard layout
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Overall survival count
sns.countplot(data=df, x="survived", ax=axes[0, 0])  # FIX: lowercase
axes[0, 0].set_title("Overall Survival Count")
axes[0, 0].set_xlabel("Survived (0 = No, 1 = Yes)")

# 2. Survival by Sex
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0, 1])  # FIX: lowercase
axes[0, 1].set_title("Survival by Sex")

# 3. Survival by Passenger Class
sns.countplot(data=df, x="pclass", hue="survived", ax=axes[0, 2])  # FIX: lowercase
axes[0, 2].set_title("Survival by Passenger Class")

# 4. Age distribution by survival
sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True, ax=axes[1, 0])  # FIX: lowercase
axes[1, 0].set_title("Age Distribution by Survival")

# 5. Fare vs Age scatter colored by survival
sns.scatterplot(data=df, x="age", y="fare", hue="survived", ax=axes[1, 1])  # FIX: lowercase
axes[1, 1].set_title("Fare vs Age by Survival")

# 6. Correlation heatmap
corr = df[["survived", "pclass", "age", "fare", "sibsp", "parch"]].corr()  # FIX: lowercase
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[1, 2])
axes[1, 2].set_title("Feature Correlation Heatmap")

plt.tight_layout()

# FIX: correct savefig path
plt.savefig('titanic_results/gpt5.2/dashboard/run3_poor_context/dashboard_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()