
from pathlib import Path
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "tourist_arrivals.csv"

OUTPUT_FILE = PROCESSED_DIR / "annual_tourist_arrivals.csv"


# ==========================================
# 2. LOAD TOURISM DATA
# ==========================================

print("\n======================================")
print(" ANNUAL TOURIST ARRIVALS PROCESSOR")
print("======================================")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Source dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nSource dataset loaded successfully!")

print("\nOriginal Dataset:")
print(df.head())


# ==========================================
# 3. VALIDATE REQUIRED COLUMNS
# ==========================================

required_columns = [
    "year",
    "domestic_tourist_visits",
    "foreign_tourist_visits"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================
# 4. DATA CLEANING
# ==========================================

print("\nCleaning tourism data...")

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(subset=required_columns)

df = df[
    (df["year"] >= 1900) &
    (df["domestic_tourist_visits"] >= 0) &
    (df["foreign_tourist_visits"] >= 0)
]

df["year"] = df["year"].astype(int)

print("Data cleaning completed!")


# ==========================================
# 5. AGGREGATE YEAR-WISE TOURIST ARRIVALS
# ==========================================

print("\nCalculating annual tourist arrivals...")

annual_df = (
    df.groupby("year", as_index=False)
    .agg({
        "domestic_tourist_visits": "sum",
        "foreign_tourist_visits": "sum"
    })
)


# ==========================================
# 6. CALCULATE TOTAL TOURIST VISITS
# ==========================================

annual_df["total_tourist_visits"] = (
    annual_df["domestic_tourist_visits"] +
    annual_df["foreign_tourist_visits"]
)


# ==========================================
# 7. SORT BY YEAR
# ==========================================

annual_df = annual_df.sort_values(
    "year"
).reset_index(drop=True)


# ==========================================
# 8. EXPORT FINAL CSV
# ==========================================

annual_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nAnnual Tourism Dataset Generated!")

print("\nFinal Annual Tourist Arrivals:")
print(annual_df.to_string(index=False))


# ==========================================
# 9. DATASET SUMMARY
# ==========================================

print("\n======================================")
print(" DATASET SUMMARY")
print("======================================")

print("Total Years:", len(annual_df))

print(
    "Total Domestic Visits:",
    f"{annual_df['domestic_tourist_visits'].sum():,}"
)

print(
    "Total Foreign Visits:",
    f"{annual_df['foreign_tourist_visits'].sum():,}"
)

print(
    "Total Tourist Visits:",
    f"{annual_df['total_tourist_visits'].sum():,}"
)

print("\nCSV Saved Successfully:")
print(OUTPUT_FILE)

print("\nAnnual Tourism Processing Completed!")
