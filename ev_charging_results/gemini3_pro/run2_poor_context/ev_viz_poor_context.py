'''
Prompt: 
The EV Charging Patterns dataset contains session-level information about electric vehicle charging behavior including energy usage, duration, and vehicle characteristics.
Create a visualization showing how different features relate to and impact energy consumption.
Provide ONLY the Python code wrapped in triple backticks.

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/ev_charging_patterns.csv')

# Select numerical columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Target column for energy consumption
target = 'Energy Consumed (kWh)'

if target in numeric_df.columns:
    # Calculate correlation of all numeric features with the target variable
    correlations = numeric_df.corr()[[target]].sort_values(by=target, ascending=False)
    
    # Drop the target itself from the rows to show only the impact of OTHER features
    correlations = correlations.drop(index=target)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=correlations[target], 
        y=correlations.index, 
        palette='coolwarm', 
        hue=correlations[target], 
        legend=False
    )
    
    # Add titles and labels
    plt.title('Impact of Features on Energy Consumed (kWh)\n(Pearson Correlation Coefficient)', fontsize=14, pad=15)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add vertical line at 0 for reference
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('ev_charging_results/gemini3_pro/run2_poor_context/ev_viz_poor_context_output.png', dpi=300, bbox_inches='tight')
    plt.close()