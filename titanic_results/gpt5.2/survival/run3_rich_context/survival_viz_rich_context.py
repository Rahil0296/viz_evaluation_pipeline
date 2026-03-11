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

# Load dataset
df = pd.read_csv('data/titanic.csv')

# Handle missing values in relevant columns
df = df.dropna(subset=['pclass', 'survived'])

# Compute survival statistics by passenger class
summary = (
    df.groupby('pclass')['survived']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'survivors', 'count': 'total'})
    .reset_index()
)

summary['survival_rate'] = summary['survivors'] / summary['total'] * 100

# Sort by class for clarity
summary = summary.sort_values('pclass')

# Set visualization style
sns.set_theme(style="whitegrid")
palette = sns.color_palette("colorblind", len(summary))

# Create bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(summary['pclass'].astype(str), summary['survival_rate'], color=palette)

# Add annotations (survivor count and percentage)
for bar, survivors, total, rate in zip(
    bars, summary['survivors'], summary['total'], summary['survival_rate']
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{survivors}/{total}\n({rate:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=10
    )

# Labels and title
plt.title("Titanic Survival Rate by Passenger Class", fontsize=14)
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.ylim(0, 100)

# Improve layout
plt.tight_layout()

# Save high-resolution output
plt.savefig('titanic_results/gpt5.2/survival/run3_rich_context/survival_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()