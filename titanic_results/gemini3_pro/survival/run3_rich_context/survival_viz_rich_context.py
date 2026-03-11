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
import os

# Define file path (handles both direct and data/ directory structures)
file_path = 'data/titanic.csv'
if not os.path.exists(file_path):
    file_path = 'titanic.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Aggregate data to get total counts, survived counts, and survival rates by class
summary = df.groupby('class').agg(
    total=('survived', 'count'),
    survived_count=('survived', 'sum'),
    survival_rate=('survived', 'mean')
).reset_index()

# Sort the summary by passenger class logically (First, Second, Third)
class_order = {'First': 1, 'Second': 2, 'Third': 3}
summary['class_rank'] = summary['class'].map(class_order)
summary = summary.sort_values('class_rank')

# Set up the plotting style and figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

# Define X-axis positions and bar width
x = range(len(summary))
width = 0.35

# Colorblind-friendly palette (Okabe-Ito hex codes)
color_total = '#0072B2'     # Blue
color_survived = '#D55E00'  # Vermillion
color_rate = '#009E73'      # Green

# Plot side-by-side bars for total passengers and survivors
bars1 = ax1.bar([i - width/2 for i in x], summary['total'], width, label='Total Passengers', color=color_total)
bars2 = ax1.bar([i + width/2 for i in x], summary['survived_count'], width, label='Survived Count', color=color_survived)

# Set primary axis labels and titles
ax1.set_xlabel('Passenger Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Passengers', fontsize=12, fontweight='bold')
ax1.set_title('Titanic Survival Rate and Counts by Passenger Class', fontsize=15, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(summary['class'], fontsize=11)

# Function to add count annotations on top of bars
def annotate_bars(bars, ax):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), 
                ha='center', va='bottom', fontsize=10)

annotate_bars(bars1, ax1)
annotate_bars(bars2, ax1)

# Create a secondary y-axis for the survival percentage line
ax2 = ax1.twinx()
ax2.plot(x, summary['survival_rate'] * 100, color=color_rate, marker='o', 
         linewidth=2, markersize=8, label='Survival Rate (%)')

# Configure secondary axis
ax2.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(False) # Turn off grid for secondary axis to avoid clutter

# Annotate percentage points on the line plot
for i, rate in enumerate(summary['survival_rate']):
    ax2.text(i, rate * 100 + 3, f'{rate*100:.1f}%', 
             ha='center', va='bottom', fontweight='bold', color=color_rate, fontsize=11)

# Combine legends from both axes into a single legend box
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, framealpha=0.9)

# Adjust layout and save the plot with high DPI
plt.tight_layout()

plt.savefig('titanic_results/gemini3_pro/survival/run3_rich_context/survival_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()