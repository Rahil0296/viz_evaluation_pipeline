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
import os

# Load data gracefully (handling both root and 'data/' directory scenarios)
file_path = 'titanic.csv'
if not os.path.exists(file_path):
    file_path = 'data/titanic.csv'

df = pd.read_csv(file_path)

# Handle missing values in the relevant columns just in case
df = df.dropna(subset=['survived', 'pclass', 'class'])

# Calculate counts and survival rates grouped by passenger class
stats = df.groupby(['pclass', 'class']).agg(
    total_passengers=('survived', 'count'),
    survived_count=('survived', 'sum')
).reset_index()

stats['survival_rate'] = (stats['survived_count'] / stats['total_passengers']) * 100
stats = stats.sort_values('pclass')  # Ensure order is 1st, 2nd, 3rd

# Set up the visualization style
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")

# Create the figure
fig, ax = plt.subplots(figsize=(9, 6))

# Use a colorblind-friendly palette from seaborn
cb_palette = sns.color_palette("colorblind")

# Plot the survival rates as a bar chart
bars = sns.barplot(
    data=stats, 
    x='class', 
    y='survival_rate', 
    palette=cb_palette[:3], 
    ax=ax,
    hue='class',
    legend=False
)

# Customize axes and titles
ax.set_ylim(0, max(stats['survival_rate']) * 1.25)  # Add space for annotations above the bars
ax.set_title('Titanic Survival Rates by Passenger Class', fontsize=16, pad=20, fontweight='bold')
ax.set_xlabel('Passenger Class', fontsize=13, labelpad=10)
ax.set_ylabel('Survival Rate (%)', fontsize=13, labelpad=10)

# Add annotations showing BOTH percentage and count of survivors
for i, row in stats.reset_index(drop=True).iterrows():
    rate = row['survival_rate']
    survivors = row['survived_count']
    total = row['total_passengers']
    
    # Text displays percentage on top, and fraction of survivors out of total below it
    label_text = f"{rate:.1f}%\n({survivors} survived)"
    
    ax.text(
        i, 
        rate + 1.5, 
        label_text, 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        fontweight='bold',
        color='#333333'
    )

# Clean up the layout
sns.despine(left=True, bottom=False)
plt.tight_layout()

# Save the final figure with high resolution
plt.savefig('titanic_results/gemini3_pro/survival/run2_rich_context/survival_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()