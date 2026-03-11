'''
Prompt : 
You are an expert data scientist and visualization specialist.

## Dataset Context:

The Titanic dataset contains passenger information from the RMS Titanic disaster of 1912.
This is a sociodemographic dataset with approximately 891 passenger records.

## Dataset Schema (IMPORTANT: All column names are lowercase):

- **survived**: Survival status (0 = No, 1 = Yes) - TARGET VARIABLE
- **pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) - Socioeconomic proxy
- **sex**: Gender (male/female)
- **age**: Age in years (float, some missing values ~20%)
- **sibsp**: Number of siblings/spouses aboard (integer)
- **parch**: Number of parents/children aboard (integer)
- **fare**: Passenger fare (float)
- **embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **class**: Passenger class text (First, Second, Third)
- **who**: Person category (man, woman, child)
- **adult_male**: Boolean for adult male
- **deck**: Deck number (many missing values ~77%)
- **embark_town**: Full embarkation city name
- **alive**: Survival text (yes/no)
- **alone**: Boolean if passenger was alone

## Technical Requirements:

- Use Python with matplotlib, seaborn, or plotly
- Load data from 'data/titanic.csv'
- Handle missing values appropriately
- Save the final visualization as 'output.png' with high DPI (300)
- Include all necessary imports
- Make the code fully executable

## Design Principles:

- **Clarity**: Use clear, readable labels and titles
- **Color Accessibility**: Use colorblind-friendly palettes
- **Data Fidelity**: Accurately represent data without distortion
- **Interpretability**: Include legends, axis labels, and annotations

## Your Task: Survival Rate by Passenger Class

Create a visualization showing survival rates across different passenger classes (1st, 2nd, 3rd).

**Specific Requirements:**

- Show both the count of survivors and survival percentage for each class
- Use a bar chart or grouped bar 
- Use colorblind-friendly colors
- Save as 'output.png' with DPI=300

Provide ONLY the Python code wrapped in triple backticks.

'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/titanic.csv')

# Handle missing values relevant to this analysis
df = df.dropna(subset=['pclass', 'survived'])

# Compute totals and survivors by passenger class
summary = (
    df.groupby('pclass')['survived']
    .agg(total_passengers='count', survivors='sum')
    .reset_index()
)

# Calculate survival percentage
summary['survival_rate'] = (summary['survivors'] / summary['total_passengers']) * 100

# Map class labels for clarity
class_labels = {1: 'First', 2: 'Second', 3: 'Third'}
summary['class_label'] = summary['pclass'].map(class_labels)

# Set visualization style
sns.set(style="whitegrid")
palette = sns.color_palette("colorblind", 3)

# Create figure
plt.figure(figsize=(8, 6))

# Bar chart showing survivor counts
ax = sns.barplot(
    data=summary,
    x='class_label',
    y='survivors',
    palette=palette
)

# Add annotations for survival percentage and counts
for i, row in summary.iterrows():
    ax.text(
        i,
        row['survivors'] + max(summary['survivors']) * 0.02,
        f"{row['survival_rate']:.1f}%\n({int(row['survivors'])} survivors)",
        ha='center',
        va='bottom',
        fontsize=10
    )

# Labels and title
ax.set_title('Titanic Survival by Passenger Class', fontsize=14)
ax.set_xlabel('Passenger Class')
ax.set_ylabel('Number of Survivors')

# Improve layout
plt.tight_layout()

# Save high-resolution output

plt.savefig('titanic_results/gpt5.2/survival/run2_rich_context/survival_viz_rich_context_output.png', dpi=300, bbox_inches='tight')

plt.close()