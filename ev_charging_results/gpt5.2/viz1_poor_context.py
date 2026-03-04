'''
Prompt: 
The EV Charging Patterns dataset contains session-level information about electric vehicle charging behavior including energy usage, duration, and vehicle characteristics.
Create a visualization showing how different features relate to and impact energy consumption.
Provide ONLY the Python code wrapped in triple backticks.

'''


import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "data/ev_charging_patterns.csv"
df = pd.read_csv(file_path)

# Create scatter plot
plt.figure()
plt.scatter(
    df["Charging Duration (hours)"],
    df["Energy Consumed (kWh)"],
    s=df["Battery Capacity (kWh)"] * 2,
    alpha=0.6
)

plt.xlabel("Charging Duration (hours)")
plt.ylabel("Energy Consumed (kWh)")
plt.title("Impact of Charging Duration and Battery Capacity on Energy Consumption")

plt.savefig('ev_charging_results/gpt5.2/viz1_poor_context_output.png', dpi=300, bbox_inches='tight')
plt.close()