'''
Prompt:

You are an expert data scientist and visualization specialist.

## Dataset Context:
The EV Charging Patterns dataset contains session-level records of electric vehicle charging behavior including energy usage, session duration, vehicle characteristics, and environmental conditions.

## Dataset Schema:
- Battery Capacity (kWh): Total battery size of the vehicle
- Charging Duration (hours): How long the session lasted
- Charging Rate (kW): Power delivered during session
- Charging Cost (USD): Total cost of the session
- State of Charge (Start %): Battery level at session start
- State of Charge (End %): Battery level at session end
- Distance Driven (since last charge) (km): Distance since previous charge
- Temperature (°C): Ambient temperature during session
- Vehicle Age (years): Age of the vehicle
- Energy Consumed (kWh): Total energy used — TARGET VARIABLE

## Technical Requirements:
- Use Python with pandas, matplotlib, or seaborn
- Load data from 'data/ev_charging_patterns.csv'
- Use ONLY the specified numeric features
- Drop rows with missing values
- Compute a correlation matrix across all specified features
- Visualize as an annotated heatmap
- Save the final visualization as 'output.png' with DPI=300
- Include all necessary imports
- Make the code fully executable

## Design Principles:
- Use a diverging colormap (e.g. coolwarm or RdBu) centered at zero
- Use readable axis labels and title
- Include a colorbar
- Annotate each cell with its correlation value rounded to 2 decimal places

## Your Task: Feature Correlation with Energy Consumption

Create a correlation heatmap showing how all features relate to each other, with particular emphasis on their relationship to Energy Consumed (kWh).

**Specific Requirements:**
- Energy Consumed (kWh) must be visible as a row and column in the heatmap
- All correlation values must be annotated inside cells
- Title must reference energy consumption

Provide ONLY the Python code wrapped in triple backticks.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/ev_charging_patterns.csv')

# Select specified numeric features
columns = [
    'Battery Capacity (kWh)',
    'Charging Duration (hours)',
    'Charging Rate (kW)',
    'Charging Cost (USD)',
    'State of Charge (Start %)',
    'State of Charge (End %)',
    'Distance Driven (since last charge) (km)',
    'Temperature (°C)',
    'Vehicle Age (years)',
    'Energy Consumed (kWh)'
]

df = df[columns]

# Drop missing values
df = df.dropna()

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    cbar=True
)

# Title and layout
plt.title('Feature Correlation Heatmap (Focus on Energy Consumption)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Save figure
plt.tight_layout()

# Show plot
plt.savefig('ev_charging_results/gpt5.2/run3_rich_context/ev_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()