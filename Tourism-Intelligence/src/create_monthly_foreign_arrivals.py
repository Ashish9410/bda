
from pathlib import Path
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "monthly_foreign_arrivals.csv"


# ==========================================
# 2. OFFICIAL TOURISM DATA
# ==========================================
# Source: India Tourism Data Compendium 2025
# Ministry of Tourism, Government of India
# Table 2.2.1 - Month-wise Foreign Tourist Arrivals
#
# Values are actual published figures.
# Years available in this table: 2019, 2023, 2024.
# These are not continuous monthly observations.

tourism_data = {
    "January":   [1111040, 891084, 981609],
    "February":  [1090516, 893352, 1032301],
    "March":     [978236, 824549, 893021],
    "April":     [774651, 625772, 672684],
    "May":       [615136, 618086, 622189],
    "June":      [726446, 667720, 722615],
    "July":      [818125, 786486, 799842],
    "August":    [800837, 664155, 661322],
    "September": [751513, 666946, 691247],
    "October":   [945017, 832233, 840330],
    "November":  [1092440, 948631, 973941],
    "December":  [1226398, 1101914, 1060621]
}

years = [2019, 2023, 2024]


# ==========================================
# 3. CONVERT DATA INTO ROWS
# ==========================================

print("\n==========================================")
print(" MONTHLY FOREIGN TOURIST ARRIVALS")
print("==========================================")

records = []

for month_number, (month, values) in enumerate(
    tourism_data.items(), start=1
):

    for year, arrivals in zip(years, values):

        records.append({
            "year": year,
            "month": month_number,
            "month_name": month,
            "foreign_tourist_arrivals": arrivals,
            "source": "India Tourism Data Compendium 2025"
        })


# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(records)

print("\nDataset created successfully!")

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))


# ==========================================
# 5. DATA VALIDATION
# ==========================================

print("\nValidating data...")

df["year"] = pd.to_numeric(
    df["year"], errors="coerce"
)

df["month"] = pd.to_numeric(
    df["month"], errors="coerce"
)

df["foreign_tourist_arrivals"] = pd.to_numeric(
    df["foreign_tourist_arrivals"],
    errors="coerce"
)

df = df.dropna(
    subset=["year", "month", "foreign_tourist_arrivals"]
)

df = df[
    (df["month"].between(1, 12)) &
    (df["foreign_tourist_arrivals"] >= 0)
]

df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["foreign_tourist_arrivals"] = (
    df["foreign_tourist_arrivals"].astype("int64")
)

df = df.drop_duplicates(
    subset=["year", "month"]
)

df = df.sort_values(
    ["year", "month"]
).reset_index(drop=True)


# ==========================================
# 6. ADD DATE COLUMN
# ==========================================

df["date"] = pd.to_datetime(
    dict(
        year=df["year"],
        month=df["month"],
        day=1
    )
)

df = df[
    [
        "date",
        "year",
        "month",
        "month_name",
        "foreign_tourist_arrivals",
        "source"
    ]
]


# ==========================================
# 7. YEAR-WISE SUMMARY
# ==========================================

print("\n========== YEAR-WISE SUMMARY ==========")

year_summary = (
    df.groupby("year")
    ["foreign_tourist_arrivals"]
    .agg(["sum", "mean", "min", "max"])
    .reset_index()
)

print(year_summary.to_string(index=False))


# ==========================================
# 8. SAVE CSV DATASET
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==========================================")
print(" DATASET GENERATED SUCCESSFULLY")
print("==========================================")

print("Total Records:", len(df))
print("Available Years:", sorted(df["year"].unique()))

print("\nCSV Location:")
print(OUTPUT_FILE)

print("\nMonthly Foreign Arrivals Processing Completed!")
