
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "output" / "analysis"
CHART_DIR = PROJECT_ROOT / "output" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. PROFESSIONAL COLOR THEME
# ==========================================

COLORS = {
    "navy": "#17365D",
    "blue": "#168AAD",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "coral": "#E76F51",
    "background": "#F4F7FB",
    "text": "#263445",
    "grid": "#E5EAF0"
}

sns.set_theme(
    style="whitegrid",
    context="notebook",
    rc={
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.7,
        "font.family": "sans-serif"
    }
)


# ==========================================
# 3. HELPER FUNCTION
# ==========================================

def save_chart(filename):

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=COLORS["background"]
    )

    plt.close()

    print(f"Chart saved: {filename}")


# ==========================================
# 4. LOAD ANALYSIS DATASETS
# ==========================================

print("\n======================================")
print(" TOURISM INTELLIGENCE VISUALIZATION")
print("======================================")

state_df = pd.read_csv(
    DATA_DIR / "state_wise_analysis.csv"
)

comparison_df = pd.read_csv(
    DATA_DIR / "tourism_comparison.csv"
)

monthly_df = pd.read_csv(
    DATA_DIR / "monthly_trends.csv"
)

hotel_df = pd.read_csv(
    DATA_DIR / "hotel_analysis.csv"
)

growth_df = pd.read_csv(
    DATA_DIR / "tourism_growth.csv"
)

destination_df = pd.read_csv(
    DATA_DIR / "destination_popularity.csv"
)

print("\nAll datasets loaded successfully!")


# ==========================================
# 5. STATE-WISE TOURIST ARRIVALS
# ==========================================

print("\nGenerating State-wise Arrivals Chart...")

state_plot = state_df.sort_values(
    "total_visitors",
    ascending=False
)

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=state_plot,
    x="state",
    y="total_visitors",
    color=COLORS["navy"],
    errorbar=None
)

plt.title(
    "State-wise Tourist Arrivals",
    fontsize=17,
    fontweight="bold",
    pad=18
)

plt.xlabel("State", fontsize=12)
plt.ylabel("Total Tourist Visits", fontsize=12)

plt.xticks(rotation=25, ha="right")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

save_chart("state_wise_arrivals.png")


# ==========================================
# 6. DOMESTIC VS FOREIGN TOURISM
# ==========================================

print("\nGenerating Tourism Comparison Chart...")

domestic = comparison_df["total_domestic"].iloc[0]
foreign = comparison_df["total_foreign"].iloc[0]

plt.figure(figsize=(8, 8))

plt.pie(
    [domestic, foreign],
    labels=[
        "Domestic Tourists",
        "Foreign Tourists"
    ],
    colors=[
        COLORS["navy"],
        COLORS["teal"]
    ],
    autopct="%1.1f%%",
    startangle=90,
    explode=(0.04, 0),
    pctdistance=0.75,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    },
    textprops={
        "fontsize": 11,
        "fontweight": "medium"
    }
)

plt.title(
    "Domestic vs Foreign Tourist Arrivals",
    fontsize=16,
    fontweight="bold",
    pad=20
)

save_chart("domestic_vs_foreign.png")


# ==========================================
# 7. MONTHLY TOURISM TRENDS
# ==========================================

print("\nGenerating Monthly Tourism Trends Chart...")

monthly_df["period"] = pd.to_datetime(
    monthly_df["year"].astype(str)
    + "-"
    + monthly_df["month"].astype(str).str.zfill(2)
)

monthly_df = monthly_df.sort_values("period")

plt.figure(figsize=(12, 6))

ax = sns.lineplot(
    data=monthly_df,
    x="period",
    y="total_arrivals",
    color=COLORS["teal"],
    marker="o",
    linewidth=2.5,
    markersize=8
)

plt.title(
    "Monthly Tourist Arrival Trends",
    fontsize=17,
    fontweight="bold",
    pad=18
)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Tourist Arrivals", fontsize=12)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

plt.xticks(rotation=35)

save_chart("monthly_tourism_trends.png")


# ==========================================
# 8. HOTEL OCCUPANCY ANALYSIS
# ==========================================

print("\nGenerating Hotel Occupancy Chart...")

hotel_plot = hotel_df.sort_values(
    "occupancy_rate_pct",
    ascending=False
)

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=hotel_plot,
    x="destination",
    y="occupancy_rate_pct",
    color=COLORS["blue"],
    errorbar=None
)

plt.title(
    "Hotel Occupancy by Destination",
    fontsize=17,
    fontweight="bold",
    pad=18
)

plt.xlabel("Destination", fontsize=12)
plt.ylabel("Occupancy Rate (%)", fontsize=12)

plt.ylim(0, 100)

plt.xticks(rotation=25, ha="right")

save_chart("hotel_occupancy.png")


# ==========================================
# 9. TOURISM GROWTH ANALYSIS
# ==========================================

print("\nGenerating Tourism Growth Chart...")

growth_plot = growth_df.dropna(
    subset=["growth_percentage"]
)

if not growth_plot.empty:

    plt.figure(figsize=(11, 6))

    ax = sns.barplot(
        data=growth_plot,
        x="state",
        y="growth_percentage",
        hue="year",
        palette=[
            COLORS["gold"],
            COLORS["coral"],
            COLORS["blue"]
        ],
        errorbar=None
    )

    plt.title(
        "Year-over-Year Tourism Growth",
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    plt.xlabel("State", fontsize=12)
    plt.ylabel("Growth (%)", fontsize=12)

    plt.xticks(rotation=25, ha="right")

    plt.legend(
        title="Year",
        frameon=True
    )

    save_chart("tourism_growth.png")

else:
    print("No sufficient yearly growth data available.")


# ==========================================
# 10. TOP TOURIST DESTINATIONS
# ==========================================

print("\nGenerating Top Destinations Chart...")

top_destinations = (
    destination_df
    .nlargest(10, "total_visitors")
    .sort_values("total_visitors", ascending=True)
)

plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=top_destinations,
    x="total_visitors",
    y="state",
    color=COLORS["teal"],
    errorbar=None
)

plt.title(
    "Top Tourist Destinations",
    fontsize=17,
    fontweight="bold",
    pad=18
)

plt.xlabel("Total Tourist Visits", fontsize=12)
plt.ylabel("State", fontsize=12)

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

save_chart("top_destinations.png")


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n======================================")
print(" VISUALIZATION COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated Charts:")

for chart in sorted(CHART_DIR.glob("*.png")):
    print(" -", chart.name)

print("\nCharts saved at:")
print(CHART_DIR)

print("\nTourism Intelligence Visualization Completed!")
