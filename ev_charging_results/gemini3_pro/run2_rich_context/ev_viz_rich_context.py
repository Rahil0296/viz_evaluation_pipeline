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
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the file path (handling both current directory and 'data/' subdirectory)
file_path = 'ev_charging_patterns.csv'
if not os.path.exists(file_path):
    file_path = 'data/ev_charging_patterns.csv'

# Load the data
df = pd.read_csv(file_path)

# Define the specified numeric features including the target variable
features = [
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

# Filter dataset to only include the specified features and drop missing values
df_numeric = df[features].dropna()

# Compute the correlation matrix
corr_matrix = df_numeric.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    center=0, 
    cbar=True,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Set the title and axis labels
plt.title('Feature Correlation Matrix Focus on Energy Consumed (kWh)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Adjust layout for readability
plt.tight_layout()

# Save the plot
plt.savefig('ev_charging_results/gemini3_pro/run2_rich_context/ev_viz_rich_context_output.png', dpi=300, bbox_inches='tight')
plt.close()