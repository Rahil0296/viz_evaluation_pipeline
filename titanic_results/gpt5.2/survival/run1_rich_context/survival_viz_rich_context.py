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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load data (required path first)
# -----------------------------
data_path = "data/titanic.csv"
fallback_path = "/mnt/data/titanic.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
elif os.path.exists(fallback_path):
    df = pd.read_csv(fallback_path)
else:
    raise FileNotFoundError("Could not find 'data/titanic.csv' or '/mnt/data/titanic.csv'.")

# -----------------------------
# Basic cleaning (handle missing)
# -----------------------------
df = df.copy()
df = df.dropna(subset=["survived", "pclass"])
df["survived"] = df["survived"].astype(int)
df["pclass"] = df["pclass"].astype(int)

# Keep only valid classes 1/2/3
df = df[df["pclass"].isin([1, 2, 3])]

# -----------------------------
# Aggregations: counts + survival %
# -----------------------------
counts = (
    df.groupby(["pclass", "survived"])
      .size()
      .reset_index(name="count")
)

# Ensure both survived categories exist for each class
all_classes = pd.Index([1, 2, 3], name="pclass")
all_surv = pd.Index([0, 1], name="survived")
full_index = pd.MultiIndex.from_product([all_classes, all_surv], names=["pclass", "survived"])

counts = (
    counts.set_index(["pclass", "survived"])
          .reindex(full_index, fill_value=0)
          .reset_index()
)

totals = counts.groupby("pclass", as_index=False)["count"].sum().rename(columns={"count": "total"})
survivors = counts[counts["survived"] == 1][["pclass", "count"]].rename(columns={"count": "survivors"})
summary = totals.merge(survivors, on="pclass", how="left").fillna({"survivors": 0})
summary["survival_rate"] = np.where(summary["total"] > 0, summary["survivors"] / summary["total"], 0.0)

# -----------------------------
# Visualization: grouped bars + annotations
# -----------------------------
sns.set_theme(style="whitegrid")

palette = sns.color_palette("colorblind", 2)  # accessible
color_not = palette[0]
color_yes = palette[1]

# Prepare plot positions
classes = [1, 2, 3]
x = np.arange(len(classes))
bar_w = 0.38

count_not = counts[(counts["pclass"].isin(classes)) & (counts["survived"] == 0)].sort_values("pclass")["count"].to_numpy()
count_yes = counts[(counts["pclass"].isin(classes)) & (counts["survived"] == 1)].sort_values("pclass")["count"].to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))

bars_not = ax.bar(x - bar_w/2, count_not, width=bar_w, label="Not Survived", color=color_not)
bars_yes = ax.bar(x + bar_w/2, count_yes, width=bar_w, label="Survived", color=color_yes)

# Axis labels / title
ax.set_title("Titanic Survival Rates by Passenger Class", fontsize=16, pad=12)
ax.set_xlabel("Passenger Class (pclass)")
ax.set_ylabel("Passenger Count")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}{'st' if c==1 else 'nd' if c==2 else 'rd'}" for c in classes])

# Add count labels on each bar
def label_bars(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{int(h)}",
            (b.get_x() + b.get_width()/2, h),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=10
        )

label_bars(bars_not)
label_bars(bars_yes)

# Add survival % + survivors/total annotation per class (above the grouped bars)
y_max = max((count_not + count_yes).max(), 1)
ax.set_ylim(0, y_max * 1.25)

summary_sorted = summary.sort_values("pclass")
for i, row in enumerate(summary_sorted.itertuples(index=False)):
    # Place annotation above the taller of the two bars for that class
    top = max(count_not[i], count_yes[i])
    pct = row.survival_rate * 100
    ax.annotate(
        f"Survival: {pct:.1f}%\n({int(row.survivors)}/{int(row.total)})",
        (x[i], top),
        textcoords="offset points",
        xytext=(0, 32),
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.85")
    )

# Legend
ax.legend(title="Outcome", frameon=True)

# Clean layout + save
plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches="tight")
plt.close(fig)
